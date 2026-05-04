//! FTS5 candidate retrieval.
//!
//! Python builds the FTS5 index at ingest time; Rust queries it at query
//! time. The on-disk format is just a SQLite file — both sides use the
//! same backend (SQLite has FTS5 built in since 3.9).
//!
//! We expect a virtual table named `reviews_fts` (matching design 05's
//! schema) with content blob in the first column. The caller can pass a
//! different table name via `table`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rusqlite::{params, Connection, OpenFlags};

/// Run an FTS5 query against `db_path` and return the top `top_k`
/// (product_id, bm25_score) pairs.
///
/// Score sign convention: SQLite's `bm25()` function returns a NEGATIVE
/// score where smaller (more negative) = better match. We flip the sign
/// before returning so callers see "higher = better" — the convention
/// every other ranker in the arena uses.
///
/// `db_path`:    path to the SQLite database
/// `query`:      the FTS5 MATCH expression. CALLER IS RESPONSIBLE FOR
///               quoting/sanitizing — see the Python wrapper, which uses
///               `tokenize_for_fts5()` to split on whitespace and quote
///               each term.
/// `top_k`:      maximum results to return
/// `table`:      FTS5 table name (default "reviews_fts")
/// `id_column`:  column holding the external product id (default "product_id")
/// `id_filter`:  optional list of product IDs to restrict to. When the
///               prefilter has already narrowed the catalog to a small
///               candidate set, passing it here avoids returning results
///               that will be discarded downstream.
#[pyfunction]
#[pyo3(signature = (db_path, query, top_k=100, table="reviews_fts", id_column="product_id", id_filter=None))]
pub fn fts5_search(
    db_path: &str,
    query: &str,
    top_k: usize,
    table: &str,
    id_column: &str,
    id_filter: Option<Vec<String>>,
) -> PyResult<Vec<(String, f64)>> {
    if !is_safe_table(table) {
        return Err(PyValueError::new_err(format!(
            "unsafe table name: {table:?}"
        )));
    }
    if !is_safe_table(id_column) {
        return Err(PyValueError::new_err(format!(
            "unsafe column name: {id_column:?}"
        )));
    }

    // Open read-only — Rust never writes the FTS index.
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("open {db_path:?}: {e}")))?;

    let mut rows: Vec<(String, f64)> = Vec::new();

    // Two SQL paths: with and without an id filter. We can't bind a
    // dynamic-length list as a single parameter; we'd need to interpolate
    // placeholders. For safety against runaway sets, cap at 5000 IDs.
    if let Some(ids) = id_filter.as_ref() {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        if ids.len() > 5000 {
            return Err(PyValueError::new_err(format!(
                "id_filter too large ({} ids); cap is 5000",
                ids.len()
            )));
        }
        let placeholders = vec!["?"; ids.len()].join(",");
        let sql = format!(
            "SELECT {id_col}, bm25({tab}) AS score \
             FROM {tab} \
             WHERE {tab} MATCH ? \
               AND {id_col} IN ({placeholders}) \
             ORDER BY score \
             LIMIT ?",
            id_col = id_column,
            tab = table,
            placeholders = placeholders,
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(format!("prepare: {e}")))?;

        // Bind: [query, id1, id2, ..., top_k]
        let mut bind_params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(ids.len() + 2);
        bind_params.push(&query);
        for id in ids {
            bind_params.push(id);
        }
        let top_k_i64 = top_k as i64;
        bind_params.push(&top_k_i64);

        let mut q = stmt
            .query(rusqlite::params_from_iter(bind_params))
            .map_err(|e| PyRuntimeError::new_err(format!("query: {e}")))?;
        while let Some(row) = q
            .next()
            .map_err(|e| PyRuntimeError::new_err(format!("row: {e}")))?
        {
            let id: String = row
                .get(0)
                .map_err(|e| PyRuntimeError::new_err(format!("col 0: {e}")))?;
            let raw_score: f64 = row
                .get(1)
                .map_err(|e| PyRuntimeError::new_err(format!("col 1: {e}")))?;
            // Flip sign — SQLite bm25 is "lower = better"; we return
            // "higher = better" so it composes with the rest of the arena.
            rows.push((id, -raw_score));
        }
    } else {
        let sql = format!(
            "SELECT {id_col}, bm25({tab}) AS score \
             FROM {tab} \
             WHERE {tab} MATCH ? \
             ORDER BY score \
             LIMIT ?",
            id_col = id_column,
            tab = table,
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(format!("prepare: {e}")))?;
        let mut q = stmt
            .query(params![query, top_k as i64])
            .map_err(|e| PyRuntimeError::new_err(format!("query: {e}")))?;
        while let Some(row) = q
            .next()
            .map_err(|e| PyRuntimeError::new_err(format!("row: {e}")))?
        {
            let id: String = row
                .get(0)
                .map_err(|e| PyRuntimeError::new_err(format!("col 0: {e}")))?;
            let raw_score: f64 = row
                .get(1)
                .map_err(|e| PyRuntimeError::new_err(format!("col 1: {e}")))?;
            rows.push((id, -raw_score));
        }
    }

    Ok(rows)
}

fn is_safe_table(s: &str) -> bool {
    if s.is_empty() || s.len() > 64 {
        return false;
    }
    let bytes = s.as_bytes();
    if !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_') {
        return false;
    }
    bytes
        .iter()
        .all(|&b| b.is_ascii_alphanumeric() || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn build_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.sqlite");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE reviews_fts USING fts5(content, product_id UNINDEXED, tokenize='porter unicode61');
             INSERT INTO reviews_fts (content, product_id) VALUES
               ('stiff race ski with edge grip and titanal construction', 'SKI-001'),
               ('powder freeride ski wide waist great float', 'SKI-002'),
               ('all mountain ski playful and forgiving', 'SKI-003'),
               ('beginner friendly soft flex easy turn', 'SKI-004');"
        ).unwrap();
        (dir, path)
    }

    #[test]
    fn basic_match_returns_hits() {
        let (_dir, path) = build_test_db();
        let hits = fts5_search(
            path.to_str().unwrap(),
            "powder",
            10,
            "reviews_fts",
            "product_id",
            None,
        )
        .unwrap();
        assert!(!hits.is_empty(), "expected matches for 'powder'");
        assert_eq!(hits[0].0, "SKI-002");
    }

    #[test]
    fn higher_is_better_after_sign_flip() {
        let (_dir, path) = build_test_db();
        let hits = fts5_search(
            path.to_str().unwrap(),
            "ski",
            10,
            "reviews_fts",
            "product_id",
            None,
        )
        .unwrap();
        // All four products contain "ski"; scores should all be positive
        // after we flip BM25's negative convention.
        assert!(hits.iter().all(|(_, s)| *s > 0.0));
        // And sorted descending.
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn id_filter_restricts_results() {
        let (_dir, path) = build_test_db();
        let restricted = vec!["SKI-001".to_string(), "SKI-003".to_string()];
        let hits = fts5_search(
            path.to_str().unwrap(),
            "ski",
            10,
            "reviews_fts",
            "product_id",
            Some(restricted),
        )
        .unwrap();
        for (id, _) in &hits {
            assert!(id == "SKI-001" || id == "SKI-003", "leaked id: {id}");
        }
    }

    #[test]
    fn top_k_truncates() {
        let (_dir, path) = build_test_db();
        let hits = fts5_search(
            path.to_str().unwrap(),
            "ski",
            2,
            "reviews_fts",
            "product_id",
            None,
        )
        .unwrap();
        assert!(hits.len() <= 2);
    }

    #[test]
    fn unsafe_table_rejected() {
        let result = fts5_search(
            "/dev/null",
            "x",
            10,
            "reviews_fts; DROP TABLE products",
            "product_id",
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn empty_id_filter_returns_empty() {
        let (_dir, path) = build_test_db();
        let hits = fts5_search(
            path.to_str().unwrap(),
            "ski",
            10,
            "reviews_fts",
            "product_id",
            Some(vec![]),
        )
        .unwrap();
        assert!(hits.is_empty());
    }
}
