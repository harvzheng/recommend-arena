//! Hard SQL prefilter assembly.
//!
//! The filter parser hands us a list of structured constraints
//! (attribute, op, value). This module turns them into a parameterized
//! SQL fragment + bound parameters that callers can feed straight into
//! their existing query.
//!
//! Why this is in Rust: the assembly is called on every query, must be
//! injection-safe (we never interpolate user input into SQL), and is
//! pure string building — exactly the kind of work that benefits from
//! Rust's compile-time discipline. Callers (Python wrappers in design
//! 14) can trust that the returned fragment has no caller-provided text
//! in it; only attribute *names* — which Python pre-validates against
//! the schema — and parameter placeholders.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// One typed value bound into the filter SQL.
#[derive(Debug, Clone)]
pub enum BoundValue {
    Text(String),
    Number(f64),
    Integer(i64),
}

/// SQL operators we accept. Anything else → ValueError.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,
    Gte,
    Lte,
    Contains,
    NotContains,
    InList,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "eq" | "=" => Ok(Op::Eq),
            "gte" | ">=" => Ok(Op::Gte),
            "lte" | "<=" => Ok(Op::Lte),
            "contains" => Ok(Op::Contains),
            "not_contains" => Ok(Op::NotContains),
            "in" => Ok(Op::InList),
            other => Err(format!("unsupported op: {other:?}")),
        }
    }
}

/// Validate an attribute name. We only allow `[a-zA-Z_][a-zA-Z0-9_]*`
/// so that interpolating it into the SQL fragment can never cause
/// injection. Caller is also expected to pre-validate against the
/// per-domain schema, but defense in depth.
fn is_safe_ident(s: &str) -> bool {
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

/// Build the prefilter SQL fragment.
///
/// `filters`: a list of dicts, each shaped like
///   `{"attribute": str, "op": str, "value": str | int | float | list}`
/// `domain_id_param_index`: the Python-side caller's domain_id parameter
///   slot. We emit `?N` style placeholders compatible with rusqlite /
///   Python's sqlite3.
///
/// Returns `(where_fragment, params)`:
/// - `where_fragment`: a SQL snippet that goes after `AND`, using
///   `EXISTS (...)` subqueries against the existing
///   `product_attributes` / `attribute_defs` schema (the schema design
///   05 already uses).
/// - `params`: the bound values, in order.
///
/// Conditions are joined with **OR** (inclusive retrieval — the
/// retrieval and ranking stages sort out which actually match best).
/// This matches design 05's relaxed-prefilter behavior, which has been
/// the most successful prefilter strategy on the existing arena.
#[pyfunction]
#[pyo3(signature = (filters))]
pub fn build_prefilter_sql(filters: &Bound<'_, PyList>) -> PyResult<(String, Vec<PyObject>)> {
    let parsed = parse_filters(filters)?;
    if parsed.is_empty() {
        return Ok((String::new(), vec![]));
    }

    let mut conditions: Vec<String> = Vec::with_capacity(parsed.len());
    let mut params: Vec<BoundValue> = Vec::new();

    for (attr, op, value) in &parsed {
        if !is_safe_ident(attr) {
            return Err(PyValueError::new_err(format!(
                "unsafe attribute identifier: {attr:?}"
            )));
        }
        match op {
            Op::Eq => {
                // Numeric eq with a small tolerance — matches design 05's behavior.
                conditions.push(
                    "EXISTS (SELECT 1 FROM product_attributes pa \
                     JOIN attribute_defs ad ON ad.id = pa.attribute_def_id \
                     WHERE pa.product_id = p.id AND ad.name = ? \
                       AND ABS(pa.value_numeric - ?) < 0.5)"
                        .to_string(),
                );
                params.push(BoundValue::Text(attr.clone()));
                params.push(value.clone());
            }
            Op::Gte => {
                conditions.push(
                    "EXISTS (SELECT 1 FROM product_attributes pa \
                     JOIN attribute_defs ad ON ad.id = pa.attribute_def_id \
                     WHERE pa.product_id = p.id AND ad.name = ? \
                       AND pa.value_numeric >= ?)"
                        .to_string(),
                );
                params.push(BoundValue::Text(attr.clone()));
                params.push(value.clone());
            }
            Op::Lte => {
                conditions.push(
                    "EXISTS (SELECT 1 FROM product_attributes pa \
                     JOIN attribute_defs ad ON ad.id = pa.attribute_def_id \
                     WHERE pa.product_id = p.id AND ad.name = ? \
                       AND pa.value_numeric <= ?)"
                        .to_string(),
                );
                params.push(BoundValue::Text(attr.clone()));
                params.push(value.clone());
            }
            Op::Contains => {
                conditions.push(
                    "EXISTS (SELECT 1 FROM product_attributes pa \
                     JOIN attribute_defs ad ON ad.id = pa.attribute_def_id \
                     WHERE pa.product_id = p.id AND ad.name = ? \
                       AND pa.value_text LIKE ?)"
                        .to_string(),
                );
                params.push(BoundValue::Text(attr.clone()));
                if let BoundValue::Text(s) = value {
                    params.push(BoundValue::Text(format!("%{s}%")));
                } else {
                    return Err(PyValueError::new_err(
                        "contains op requires a text value",
                    ));
                }
            }
            Op::NotContains => {
                conditions.push(
                    "NOT EXISTS (SELECT 1 FROM product_attributes pa \
                     JOIN attribute_defs ad ON ad.id = pa.attribute_def_id \
                     WHERE pa.product_id = p.id AND ad.name = ? \
                       AND pa.value_text LIKE ?)"
                        .to_string(),
                );
                params.push(BoundValue::Text(attr.clone()));
                if let BoundValue::Text(s) = value {
                    params.push(BoundValue::Text(format!("%{s}%")));
                } else {
                    return Err(PyValueError::new_err(
                        "not_contains op requires a text value",
                    ));
                }
            }
            Op::InList => {
                // The "value" for an in-list op is a comma-separated text list,
                // pre-flattened by the caller. We expose IN as a series of OR
                // EXISTS subqueries to keep the fragment shape uniform.
                if let BoundValue::Text(s) = value {
                    let parts: Vec<&str> = s.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
                    if parts.is_empty() {
                        continue;
                    }
                    let mut sub: Vec<String> = Vec::with_capacity(parts.len());
                    for part in parts {
                        sub.push(
                            "EXISTS (SELECT 1 FROM product_attributes pa \
                             JOIN attribute_defs ad ON ad.id = pa.attribute_def_id \
                             WHERE pa.product_id = p.id AND ad.name = ? \
                               AND pa.value_text LIKE ?)"
                                .to_string(),
                        );
                        params.push(BoundValue::Text(attr.clone()));
                        params.push(BoundValue::Text(format!("%{part}%")));
                    }
                    conditions.push(format!("({})", sub.join(" OR ")));
                } else {
                    return Err(PyValueError::new_err("in op requires a text value"));
                }
            }
        }
    }

    let where_fragment = format!("({})", conditions.join(" OR "));

    // Convert BoundValues to PyObjects so the Python caller can pass them
    // straight into sqlite3.execute(...).
    Python::with_gil(|py| {
        let py_params: Vec<PyObject> = params
            .into_iter()
            .map(|v| match v {
                BoundValue::Text(s) => s.into_py(py),
                BoundValue::Number(n) => n.into_py(py),
                BoundValue::Integer(i) => i.into_py(py),
            })
            .collect();
        Ok((where_fragment, py_params))
    })
}

fn parse_filters(filters: &Bound<'_, PyList>) -> PyResult<Vec<(String, Op, BoundValue)>> {
    let mut out = Vec::with_capacity(filters.len());
    for item in filters.iter() {
        let dict: &Bound<'_, PyDict> = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each filter must be a dict"))?;

        let attr: String = dict
            .get_item("attribute")?
            .ok_or_else(|| PyValueError::new_err("filter missing 'attribute'"))?
            .extract()?;
        let op_s: String = dict
            .get_item("op")?
            .ok_or_else(|| PyValueError::new_err("filter missing 'op'"))?
            .extract()?;
        let op = Op::parse(&op_s).map_err(PyValueError::new_err)?;
        let value_obj = dict
            .get_item("value")?
            .ok_or_else(|| PyValueError::new_err("filter missing 'value'"))?;

        let value = if let Ok(s) = value_obj.extract::<String>() {
            BoundValue::Text(s)
        } else if let Ok(i) = value_obj.extract::<i64>() {
            BoundValue::Integer(i)
        } else if let Ok(f) = value_obj.extract::<f64>() {
            BoundValue::Number(f)
        } else if let Ok(seq) = value_obj.extract::<Vec<String>>() {
            // Flatten string sequences into a comma-joined text value
            // for the InList op. Numeric sequences not supported here —
            // the schema doesn't need them yet.
            BoundValue::Text(seq.join(","))
        } else {
            return Err(PyValueError::new_err(
                "filter value must be str, int, float, or list[str]",
            ));
        };

        out.push((attr, op, value));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_ident_accepts_normal_names() {
        assert!(is_safe_ident("waist_width_mm"));
        assert!(is_safe_ident("terrain"));
        assert!(is_safe_ident("_private"));
        assert!(is_safe_ident("a"));
    }

    #[test]
    fn safe_ident_rejects_injection_attempts() {
        assert!(!is_safe_ident(""));
        assert!(!is_safe_ident("'; DROP TABLE products; --"));
        assert!(!is_safe_ident("col WITH space"));
        assert!(!is_safe_ident("1col"));
        assert!(!is_safe_ident("col-name"));
        assert!(!is_safe_ident("col)"));
    }

    #[test]
    fn op_parsing() {
        assert_eq!(Op::parse("gte").unwrap(), Op::Gte);
        assert_eq!(Op::parse(">=").unwrap(), Op::Gte);
        assert_eq!(Op::parse("contains").unwrap(), Op::Contains);
        assert!(Op::parse("DROP TABLE").is_err());
    }
}
