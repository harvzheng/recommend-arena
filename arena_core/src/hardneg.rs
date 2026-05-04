//! Hard-negative mining for the synthetic query generator.
//!
//! Given a query vector and a pool of candidate vectors (one per
//! product), pick the `k` candidates that are MOST similar to the
//! query but NOT in the positive set. These are the hard negatives
//! used in contrastive fine-tuning (MultipleNegativesRankingLoss).
//!
//! Why Rust: this is called once per generated query, and the pool
//! grows linearly with the catalog. For 10K products × 10K queries,
//! the Python+NumPy version costs ~minutes on warm disk; this Rust
//! version is ~10x faster and bounded by memory bandwidth instead of
//! per-call NumPy overhead.
//!
//! Vectors are assumed to be L2-normalized (sentence-transformers'
//! default). Cosine similarity is then a dot product.

use pyo3::prelude::*;
use std::collections::HashSet;

/// Pick the top `k` hardest negatives for a query.
///
/// `query_vec`:    the query embedding (L2-normalized).
/// `candidate_ids`: product IDs in the catalog, parallel to `candidate_vecs`.
/// `candidate_vecs`: flat row-major matrix of shape `(N, dim)`.
/// `dim`:          embedding dimension.
/// `positive_id`:  product id that's the *correct* answer for this query;
///                 always excluded.
/// `also_exclude`: any other ids to keep out of the negative pool.
/// `k`:            number of negatives to return.
///
/// Returns the chosen ids in descending similarity order (hardest first).
#[pyfunction]
#[pyo3(signature = (
    query_vec, candidate_ids, candidate_vecs, dim,
    positive_id, also_exclude=Vec::new(), k=5
))]
pub fn hard_negative_mine(
    query_vec: Vec<f32>,
    candidate_ids: Vec<String>,
    candidate_vecs: Vec<f32>,
    dim: usize,
    positive_id: &str,
    also_exclude: Vec<String>,
    k: usize,
) -> PyResult<Vec<String>> {
    if query_vec.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "query_vec has {} dims, expected {}",
            query_vec.len(),
            dim
        )));
    }
    if candidate_ids.len() * dim != candidate_vecs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "candidate_vecs has {} floats; expected {} ({} ids × {} dim)",
            candidate_vecs.len(),
            candidate_ids.len() * dim,
            candidate_ids.len(),
            dim
        )));
    }
    Ok(mine(&query_vec, &candidate_ids, &candidate_vecs, dim, positive_id, &also_exclude, k))
}

/// Pure-Rust core (no PyO3 types) so it's testable from cargo test.
pub fn mine(
    query_vec: &[f32],
    candidate_ids: &[String],
    candidate_vecs: &[f32],
    dim: usize,
    positive_id: &str,
    also_exclude: &[String],
    k: usize,
) -> Vec<String> {
    // Build the exclusion set once.
    let mut excluded: HashSet<&str> = HashSet::with_capacity(also_exclude.len() + 1);
    excluded.insert(positive_id);
    for s in also_exclude {
        excluded.insert(s.as_str());
    }

    // Score every candidate that isn't excluded. We use a small
    // partial-sort: keep the top-k seen so far. For k << N this is
    // O(N log k) which beats sorting the full vector.
    //
    // We use a simple Vec + binary search insertion instead of
    // BinaryHeap so we can break ties by the candidate's catalog
    // position — the smaller index wins on ties for determinism.
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);

    for (i, id) in candidate_ids.iter().enumerate() {
        if excluded.contains(id.as_str()) {
            continue;
        }
        let start = i * dim;
        let cv = &candidate_vecs[start..start + dim];
        // Cosine sim on L2-normalized vectors == dot product.
        let mut score = 0.0f32;
        for (a, b) in query_vec.iter().zip(cv.iter()) {
            score += a * b;
        }

        if top.len() < k {
            top.push((score, i));
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1)));
        } else if score > top.last().unwrap().0 {
            top.pop();
            top.push((score, i));
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1)));
        }
    }

    top.into_iter()
        .map(|(_score, idx)| candidate_ids[idx].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn excludes_positive() {
        let q = vec![1.0, 0.0];
        let cv = vec![
            1.0, 0.0,  // a — the positive (most similar by far)
            0.9, 0.1,  // b
            0.5, 0.5,  // c
            0.0, 1.0,  // d
        ];
        let result = mine(&q, &ids(&["a", "b", "c", "d"]), &cv, 2, "a", &[], 2);
        assert!(!result.contains(&"a".to_string()), "positive leaked into hardneg");
        assert_eq!(result, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn respects_also_exclude() {
        let q = vec![1.0, 0.0];
        let cv = vec![
            1.0, 0.0,
            0.9, 0.1,
            0.5, 0.5,
            0.0, 1.0,
        ];
        let result = mine(
            &q,
            &ids(&["a", "b", "c", "d"]),
            &cv,
            2,
            "a",
            &["b".to_string()],
            2,
        );
        assert_eq!(result, vec!["c".to_string(), "d".to_string()]);
    }

    #[test]
    fn returns_at_most_k() {
        let q = vec![1.0, 0.0];
        let cv = vec![
            1.0, 0.0,
            0.9, 0.1,
            0.5, 0.5,
        ];
        let result = mine(&q, &ids(&["a", "b", "c"]), &cv, 2, "a", &[], 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn handles_empty_pool() {
        let q = vec![1.0, 0.0];
        let result = mine(&q, &[], &[], 2, "a", &[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn deterministic_tie_break() {
        // Two products tie exactly; the one earlier in the catalog wins.
        let q = vec![1.0, 0.0];
        let cv = vec![
            1.0, 0.0,  // positive
            0.7, 0.7,  // tie 1 (b)
            0.7, 0.7,  // tie 2 (c)  — same score, later index, loses
        ];
        let r1 = mine(&q, &ids(&["a", "b", "c"]), &cv, 2, "a", &[], 1);
        let r2 = mine(&q, &ids(&["a", "b", "c"]), &cv, 2, "a", &[], 1);
        assert_eq!(r1, r2);
        assert_eq!(r1, vec!["b".to_string()]);
    }
}
