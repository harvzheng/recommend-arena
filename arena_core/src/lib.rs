//! arena_core — hot-path Rust kernel for the recommend-arena framework.
//!
//! Exposes three Python-callable surfaces via PyO3:
//!
//! * `rrf_fuse`  — Reciprocal Rank Fusion (the design-4 score-fusion fix)
//! * `build_prefilter_sql` — assemble the hard-filter SQL fragment + params
//! * `fts5_search` — run a BM25-weighted FTS5 query against an on-disk DB
//!
//! Python owns ingestion (rare). Rust owns the per-query hot path.

use pyo3::prelude::*;

pub mod fts5;
pub mod hardneg;
pub mod prefilter;
pub mod rrf;

/// Python module entrypoint. Symbol must match `[lib] name` in Cargo.toml
/// AND `[tool.maturin] module-name` in pyproject.toml — otherwise import
/// fails silently in the wrong way.
#[pymodule]
fn arena_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rrf::rrf_fuse, m)?)?;
    m.add_function(wrap_pyfunction!(prefilter::build_prefilter_sql, m)?)?;
    m.add_function(wrap_pyfunction!(fts5::fts5_search, m)?)?;
    m.add_function(wrap_pyfunction!(hardneg::hard_negative_mine, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
