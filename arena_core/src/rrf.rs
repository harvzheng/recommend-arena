//! Reciprocal Rank Fusion.
//!
//! Why this lives here, in Rust: the design-4 failure ("naive score fusion
//! failed", NDCG@5 = 0.323) was caused by combining incommensurate score
//! distributions across retrievers. RRF only sees ranks. It is the
//! single-call hot path for every query in design 14, and pure math, so
//! it's the smallest function with the highest cumulative latency win.
//!
//! The classic formula (Cormack et al., 2009):
//!     RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))
//!
//! Items missing from a retriever's list contribute zero from that
//! retriever (i.e. we treat them as "rank infinity"). `k=60` is the
//! standard published constant.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Fuse multiple ranked lists of document IDs via RRF.
///
/// `ranked_lists`: a list of lists. Each inner list is one retriever's
///                 ranking, best-first. Duplicate IDs within a single
///                 inner list use their FIRST appearance only.
/// `k`:            RRF dampening constant (default 60).
/// `top_k`:        truncate the fused result to this many items
///                 (None → return all).
///
/// Returns a list of (id, fused_score) tuples sorted by score descending.
/// Score ties are broken by the order the ID first appeared across all
/// input lists — deterministic so tests don't flake.
#[pyfunction]
#[pyo3(signature = (ranked_lists, k=60, top_k=None))]
pub fn rrf_fuse(
    ranked_lists: Vec<Vec<String>>,
    k: u32,
    top_k: Option<usize>,
) -> Vec<(String, f64)> {
    fuse(&ranked_lists, k, top_k)
}

/// Pure-Rust core, broken out so cargo tests don't need pyo3.
pub fn fuse(
    ranked_lists: &[Vec<String>],
    k: u32,
    top_k: Option<usize>,
) -> Vec<(String, f64)> {
    let k_f = f64::from(k);
    let mut scores: HashMap<&str, f64> = HashMap::new();
    // Track first-appearance order for deterministic tie-breaking.
    let mut first_seen: HashMap<&str, usize> = HashMap::new();
    let mut counter: usize = 0;

    for list in ranked_lists {
        let mut seen_in_list: HashMap<&str, ()> = HashMap::new();
        for (rank_zero_indexed, id) in list.iter().enumerate() {
            // Skip duplicates within the same retriever's list.
            if seen_in_list.contains_key(id.as_str()) {
                continue;
            }
            seen_in_list.insert(id.as_str(), ());

            let rank = rank_zero_indexed as f64 + 1.0;
            *scores.entry(id.as_str()).or_insert(0.0) += 1.0 / (k_f + rank);

            first_seen.entry(id.as_str()).or_insert_with(|| {
                counter += 1;
                counter
            });
        }
    }

    let mut pairs: Vec<(String, f64)> = scores
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect();

    pairs.sort_by(|a, b| {
        // Higher score first; ties broken by first-appearance order.
        match b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => first_seen
                .get(a.0.as_str())
                .copied()
                .unwrap_or(usize::MAX)
                .cmp(
                    &first_seen
                        .get(b.0.as_str())
                        .copied()
                        .unwrap_or(usize::MAX),
                ),
            other => other,
        }
    });

    if let Some(n) = top_k {
        pairs.truncate(n);
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(pairs: &[(String, f64)]) -> Vec<&str> {
        pairs.iter().map(|(id, _)| id.as_str()).collect()
    }

    #[test]
    fn single_list_preserves_order() {
        let lists = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let out = fuse(&lists, 60, None);
        assert_eq!(ids(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn fully_agreeing_lists_keep_consensus_order() {
        let lists = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        ];
        let out = fuse(&lists, 60, None);
        assert_eq!(ids(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn item_in_both_lists_outranks_item_in_only_one() {
        // x is rank 1 in list A and rank 3 in list B.
        // y is rank 1 in list B only.
        // x should win because it's in both.
        let lists = vec![
            vec!["x".to_string(), "p".to_string(), "q".to_string()],
            vec!["y".to_string(), "z".to_string(), "x".to_string()],
        ];
        let out = fuse(&lists, 60, None);
        let order = ids(&out);
        let x_pos = order.iter().position(|s| *s == "x").unwrap();
        let y_pos = order.iter().position(|s| *s == "y").unwrap();
        assert!(x_pos < y_pos, "x should outrank y, got {:?}", order);
    }

    #[test]
    fn the_design_4_fix() {
        // The whole point of this module: when retriever scores live on
        // wildly different scales, naive fusion (the design-4 failure
        // mode) is dominated by whichever scale is larger. RRF sees only
        // ranks. The output below should match the rank-based answer
        // regardless of what the underlying scores were.
        let bm25_top10 = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let vec_top10 = vec!["doc3", "doc2", "doc6", "doc1", "doc7"];
        let lists: Vec<Vec<String>> = vec![
            bm25_top10.iter().map(|s| s.to_string()).collect(),
            vec_top10.iter().map(|s| s.to_string()).collect(),
        ];
        let out = fuse(&lists, 60, Some(3));
        let order = ids(&out);
        // doc2 and doc3 are in both lists' top-2 → should top the fusion.
        assert!(order.contains(&"doc2"));
        assert!(order.contains(&"doc3"));
    }

    #[test]
    fn duplicates_within_a_single_list_count_once() {
        let lists = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
        ]];
        let out = fuse(&lists, 60, None);
        assert_eq!(out.len(), 2);
        assert_eq!(ids(&out), vec!["a", "b"]);
    }

    #[test]
    fn top_k_truncates() {
        let lists = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ]];
        let out = fuse(&lists, 60, Some(2));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn empty_input_is_empty_output() {
        let lists: Vec<Vec<String>> = vec![];
        assert!(fuse(&lists, 60, None).is_empty());
        let lists: Vec<Vec<String>> = vec![vec![], vec![]];
        assert!(fuse(&lists, 60, None).is_empty());
    }

    #[test]
    fn score_sanity() {
        // For k=60, an item at rank 1 in two lists has score
        // 2 * (1 / (60 + 1)) = 2/61 ≈ 0.0328. Verify formula.
        let lists = vec![
            vec!["a".to_string()],
            vec!["a".to_string()],
        ];
        let out = fuse(&lists, 60, None);
        assert_eq!(out.len(), 1);
        let expected = 2.0 / 61.0;
        assert!(
            (out[0].1 - expected).abs() < 1e-9,
            "expected {}, got {}",
            expected,
            out[0].1
        );
    }

    #[test]
    fn deterministic_tie_break() {
        // Two items in two distinct lists — same scores. Order should be
        // stable across runs.
        let lists = vec![
            vec!["a".to_string()],
            vec!["b".to_string()],
        ];
        let out1 = fuse(&lists, 60, None);
        let out2 = fuse(&lists, 60, None);
        assert_eq!(out1, out2);
        // First-appearance order: a comes before b across the input.
        assert_eq!(out1[0].0, "a");
    }
}
