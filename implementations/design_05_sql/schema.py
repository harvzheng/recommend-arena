"""SQLite schema creation for Design 05: SQL-First / SQLite + FTS5."""

import sqlite3

SCHEMA_SQL = """
-- Core entities
CREATE TABLE IF NOT EXISTS domains (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id),
    external_id TEXT NOT NULL,
    name TEXT NOT NULL,
    brand TEXT,
    category TEXT,
    raw_specs TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(domain_id, external_id)
);

-- Attribute system (domain-specific, schema-driven)
CREATE TABLE IF NOT EXISTS attribute_defs (
    id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id),
    name TEXT NOT NULL,
    data_type TEXT NOT NULL,
    scale_min REAL,
    scale_max REAL,
    allowed_values TEXT,
    UNIQUE(domain_id, name)
);

CREATE TABLE IF NOT EXISTS product_attributes (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    attribute_def_id INTEGER NOT NULL REFERENCES attribute_defs(id),
    value_numeric REAL,
    value_text TEXT,
    confidence REAL DEFAULT 1.0,
    source TEXT,
    UNIQUE(product_id, attribute_def_id)
);

-- Reviews and full-text search
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    source TEXT,
    author TEXT,
    content TEXT NOT NULL,
    rating REAL,
    sentiment REAL DEFAULT 0.0,
    created_at TEXT
);

-- Synonym expansion table
CREATE TABLE IF NOT EXISTS synonyms (
    id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id),
    canonical TEXT NOT NULL,
    variant TEXT NOT NULL,
    UNIQUE(domain_id, canonical, variant)
);
"""

FTS_SCHEMA_SQL = """
-- FTS5 virtual table for review content (standalone, not content-synced)
CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts USING fts5(
    content,
    product_name,
    tokenize='porter unicode61'
);
"""

# Domain-specific attribute definitions
DOMAIN_ATTRIBUTE_DEFS = {
    "ski": [
        ("stiffness", "scale", 1, 10, None),
        ("damp", "scale", 1, 10, None),
        ("edge_grip", "scale", 1, 10, None),
        ("stability_at_speed", "scale", 1, 10, None),
        ("playfulness", "scale", 1, 10, None),
        ("powder_float", "scale", 1, 10, None),
        ("forgiveness", "scale", 1, 10, None),
        ("terrain", "categorical", None, None,
         '["on-piste","off-piste","all-mountain","park","touring","race","carving","groomed","freeride","powder","big-mountain","backcountry"]'),
        ("waist_width_mm", "numeric", None, None, None),
        ("turn_radius_m", "numeric", None, None, None),
        ("weight_g", "numeric", None, None, None),
        ("lengths_available", "text", None, None, None),
        ("rocker_profile", "categorical", None, None,
         '["camber","rocker_camber","rocker_camber_rocker","full_rocker","camber_rocker_tip"]'),
        ("construction", "text", None, None, None),
    ],
    "running_shoe": [
        ("cushioning", "scale", 1, 10, None),
        ("responsiveness", "scale", 1, 10, None),
        ("stability", "scale", 1, 10, None),
        ("grip", "scale", 1, 10, None),
        ("breathability", "scale", 1, 10, None),
        ("durability", "scale", 1, 10, None),
        ("weight_feel", "scale", 1, 10, None),
        ("weight_g", "numeric", None, None, None),
        ("heel_drop_mm", "numeric", None, None, None),
        ("stack_height_mm", "numeric", None, None, None),
        ("surface", "categorical", None, None,
         '["road","trail","track","treadmill"]'),
    ],
}


def init_db(db: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    db.executescript(SCHEMA_SQL)
    # FTS5 needs separate execution since IF NOT EXISTS works differently
    try:
        db.executescript(FTS_SCHEMA_SQL)
    except sqlite3.OperationalError:
        pass  # Table already exists
    db.commit()


def ensure_domain(db: sqlite3.Connection, domain: str) -> int:
    """Ensure a domain exists, creating it and its attribute defs if needed.

    Returns the domain_id.
    """
    row = db.execute(
        "SELECT id FROM domains WHERE name = ?", (domain,)
    ).fetchone()
    if row:
        return row[0]

    db.execute("INSERT INTO domains (name) VALUES (?)", (domain,))
    domain_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Seed attribute definitions
    attr_defs = DOMAIN_ATTRIBUTE_DEFS.get(domain, [])
    for name, dtype, smin, smax, allowed in attr_defs:
        db.execute(
            "INSERT OR IGNORE INTO attribute_defs "
            "(domain_id, name, data_type, scale_min, scale_max, allowed_values) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (domain_id, name, dtype, smin, smax, allowed),
        )

    db.commit()
    return domain_id
