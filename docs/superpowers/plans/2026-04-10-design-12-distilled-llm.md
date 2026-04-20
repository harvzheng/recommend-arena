# Design #12: Distilled LLM Ranker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a distilled LLM ranker that uses a teacher model (Claude/GPT-4o) to label query-product pairs, then fine-tunes a Qwen 2.5 student model via LoRA to reproduce those judgments locally.

**Architecture:** Teacher LLM scores all (query, product) pairs with structured JSON output. A LoRA-tuned Qwen 2.5 (0.5B/1.5B) student learns to reproduce teacher judgments. At query time, the student runs locally — scoring each candidate and generating natural language explanations. Product contexts are pre-built from reviews + specs and stored in SQLite.

**Tech Stack:** unsloth, peft, transformers, trl, datasets, torch, sqlite3, pytest

---

## File Structure

```
implementations/design_12_distilled_llm/
├── __init__.py           # factory: create_recommender()
├── requirements.txt      # unsloth, peft, transformers, trl, datasets, torch
├── context.py            # ProductContext building from reviews + specs
├── teacher.py            # Teacher labeling pipeline (prompt, scoring, caching)
├── dataset.py            # Convert teacher judgments to instruction-tuning format
├── student.py            # LoRA training via unsloth, export options
├── inference.py          # Student model inference + JSON parsing
├── recommender.py        # Main class implementing Recommender protocol
├── db.py                 # SQLite schema init + helpers
└── tests/
    ├── __init__.py
    ├── test_context.py   # Test context construction
    ├── test_teacher.py   # Test teacher prompt + parsing (mock LLM)
    ├── test_dataset.py   # Test training data format
    ├── test_inference.py # Test JSON extraction + fallback
    └── test_recommender.py  # Integration test
```

---

### Task 1: Database Schema (`db.py`)

**Files:**
- Create: `implementations/design_12_distilled_llm/db.py`
- Create: `implementations/design_12_distilled_llm/tests/__init__.py`

- [ ] **Step 1.1: Create `db.py` with `init_db()` and table definitions**

```python
# implementations/design_12_distilled_llm/db.py
"""SQLite schema and helpers for Design #12."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def init_db(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Create or open the SQLite database and ensure all tables exist.

    Args:
        db_path: Path to the database file, or ":memory:" for in-memory.

    Returns:
        An open sqlite3.Connection with WAL mode enabled.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS product_contexts (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    context_text TEXT NOT NULL,
    spec_summary TEXT,
    review_summary TEXT,
    review_count INTEGER DEFAULT 0,
    metadata_json TEXT,
    built_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teacher_judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    product_id TEXT NOT NULL,
    score REAL NOT NULL,
    explanation TEXT NOT NULL,
    matched_attributes_json TEXT NOT NULL,
    teacher_model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(query, product_id, teacher_model)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model TEXT NOT NULL,
    lora_rank INTEGER NOT NULL,
    num_examples INTEGER NOT NULL,
    num_epochs INTEGER NOT NULL,
    final_loss REAL,
    adapter_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_judgments_query ON teacher_judgments(query);
CREATE INDEX IF NOT EXISTS idx_judgments_product ON teacher_judgments(product_id);
CREATE INDEX IF NOT EXISTS idx_contexts_domain ON product_contexts(domain);
"""


def upsert_product_context(
    conn: sqlite3.Connection,
    *,
    product_id: str,
    product_name: str,
    domain: str,
    context_text: str,
    spec_summary: str,
    review_summary: str,
    review_count: int,
    metadata: dict,
    built_at: str,
) -> None:
    """Insert or replace a product context row."""
    conn.execute(
        "INSERT OR REPLACE INTO product_contexts "
        "(product_id, product_name, domain, context_text, "
        "spec_summary, review_summary, review_count, metadata_json, built_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            product_id, product_name, domain, context_text,
            spec_summary, review_summary, review_count,
            json.dumps(metadata), built_at,
        ),
    )


def insert_teacher_judgment(
    conn: sqlite3.Connection,
    *,
    query: str,
    product_id: str,
    score: float,
    explanation: str,
    matched_attributes: dict[str, float],
    teacher_model: str,
    created_at: str,
) -> None:
    """Insert a teacher judgment, ignoring duplicates."""
    conn.execute(
        "INSERT OR IGNORE INTO teacher_judgments "
        "(query, product_id, score, explanation, "
        "matched_attributes_json, teacher_model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            query, product_id, score, explanation,
            json.dumps(matched_attributes), teacher_model, created_at,
        ),
    )


def get_cached_judgment(
    conn: sqlite3.Connection,
    query: str,
    product_id: str,
) -> dict | None:
    """Return a cached teacher judgment or None."""
    row = conn.execute(
        "SELECT score, explanation, matched_attributes_json "
        "FROM teacher_judgments WHERE query = ? AND product_id = ?",
        (query, product_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "score": row["score"],
        "explanation": row["explanation"],
        "matched_attributes": json.loads(row["matched_attributes_json"]),
    }


def insert_training_run(
    conn: sqlite3.Connection,
    *,
    base_model: str,
    lora_rank: int,
    num_examples: int,
    num_epochs: int,
    final_loss: float | None,
    adapter_path: str,
    created_at: str,
) -> int:
    """Record a training run and return its id."""
    cur = conn.execute(
        "INSERT INTO training_runs "
        "(base_model, lora_rank, num_examples, num_epochs, "
        "final_loss, adapter_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            base_model, lora_rank, num_examples, num_epochs,
            final_loss, adapter_path, created_at,
        ),
    )
    return cur.lastrowid
```

- [ ] **Step 1.2: Create empty `tests/__init__.py`**

```python
# implementations/design_12_distilled_llm/tests/__init__.py
```

- [ ] **Step 1.3: Write `tests/test_db.py` to verify schema creation and helpers**

```python
# implementations/design_12_distilled_llm/tests/test_db.py
"""Tests for db.py — schema creation and helper functions."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.db import (
    get_cached_judgment,
    init_db,
    insert_teacher_judgment,
    insert_training_run,
    upsert_product_context,
)


def test_init_db_creates_tables():
    conn = init_db(":memory:")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "product_contexts" in tables
    assert "teacher_judgments" in tables
    assert "training_runs" in tables
    conn.close()


def test_upsert_product_context():
    conn = init_db(":memory:")
    upsert_product_context(
        conn,
        product_id="SKI-001",
        product_name="Test Ski",
        domain="ski",
        context_text="Specs: Length 170cm\n\nReviews: Great ski.",
        spec_summary="Length 170cm",
        review_summary="Great ski.",
        review_count=3,
        metadata={"length_cm": 170},
        built_at="2026-01-01T00:00:00",
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM product_contexts WHERE product_id = 'SKI-001'"
    ).fetchone()
    assert row is not None
    assert row["product_name"] == "Test Ski"
    assert row["review_count"] == 3
    conn.close()


def test_insert_and_cache_teacher_judgment():
    conn = init_db(":memory:")
    insert_teacher_judgment(
        conn,
        query="stiff carving ski",
        product_id="SKI-001",
        score=0.85,
        explanation="Strong match on stiffness.",
        matched_attributes={"stiffness": 0.9, "edge_grip": 0.8},
        teacher_model="claude-sonnet-4-20250514",
        created_at="2026-01-01T00:00:00",
    )
    conn.commit()

    cached = get_cached_judgment(conn, "stiff carving ski", "SKI-001")
    assert cached is not None
    assert cached["score"] == 0.85
    assert cached["matched_attributes"]["stiffness"] == 0.9

    miss = get_cached_judgment(conn, "powder ski", "SKI-001")
    assert miss is None
    conn.close()


def test_insert_training_run():
    conn = init_db(":memory:")
    run_id = insert_training_run(
        conn,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        lora_rank=16,
        num_examples=500,
        num_epochs=3,
        final_loss=0.42,
        adapter_path="adapters/design-12",
        created_at="2026-01-01T00:00:00",
    )
    conn.commit()
    assert run_id is not None
    row = conn.execute(
        "SELECT * FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert row["final_loss"] == 0.42
    conn.close()
```

- [ ] **Step 1.4: Run `pytest implementations/design_12_distilled_llm/tests/test_db.py -v` — verify all pass**

---

### Task 2: Context Builder (`context.py`)

**Files:**
- Create: `implementations/design_12_distilled_llm/context.py`
- Create: `implementations/design_12_distilled_llm/tests/test_context.py`

- [ ] **Step 2.1: Create `context.py` with `ProductContext` dataclass and builder functions**

```python
# implementations/design_12_distilled_llm/context.py
"""Product context construction from reviews + specs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_provider import LLMProvider


@dataclass
class ProductContext:
    """Pre-built text representation of a product for model input."""

    product_id: str
    product_name: str
    domain: str
    context_text: str
    spec_summary: str
    review_summary: str
    review_count: int
    metadata: dict = field(default_factory=dict)


_REVIEW_SUMMARY_PROMPT = (
    'Summarize what reviewers say about this {domain} product: '
    '"{product_name}".\n\n'
    'Reviews:\n{review_block}\n\n'
    'Write a 3-5 sentence summary covering the key attributes '
    'reviewers mention (performance, feel, strengths, weaknesses). '
    'Be specific -- use the reviewers\' language.'
)


def build_spec_summary(metadata: dict) -> str:
    """Format raw spec metadata as a readable one-line summary.

    Args:
        metadata: Dict of spec key-value pairs from product data.

    Returns:
        Formatted string like "Length: 177cm, Waist Width: 100mm, ...".
        Returns "No specs available." if metadata is empty.
    """
    if not metadata:
        return "No specs available."
    parts = []
    for key, value in metadata.items():
        label = key.replace("_", " ").title()
        parts.append(f"{label}: {value}")
    return ", ".join(parts)


def build_product_context(
    product: dict,
    reviews: list[dict],
    llm: LLMProvider,
    domain: str,
) -> ProductContext:
    """Build a text context for a product from its specs and reviews.

    Args:
        product: Product dict with keys "id"/"product_id", "name"/"product_name",
                 and optionally "specs"/"metadata".
        reviews: List of review dicts with "text"/"review_text" keys.
        llm: LLM provider for review summarization.
        domain: Product domain (e.g. "ski").

    Returns:
        A ProductContext with assembled context_text.
    """
    pid = product.get("product_id", product.get("id", ""))
    pname = product.get("product_name", product.get("name", ""))
    metadata = product.get("metadata", product.get("specs", {}))

    spec_summary = build_spec_summary(metadata)

    review_texts = [
        r.get("review_text", r.get("text", "")) for r in reviews
    ]
    review_texts = [t for t in review_texts if t]

    if review_texts:
        review_block = "\n---\n".join(review_texts[:20])
        prompt = _REVIEW_SUMMARY_PROMPT.format(
            domain=domain,
            product_name=pname,
            review_block=review_block,
        )
        review_summary = llm.generate(prompt)
    else:
        review_summary = "No reviews available."

    context_text = (
        f"Specs: {spec_summary}\n\n"
        f"Review consensus ({len(review_texts)} reviews): {review_summary}"
    )

    return ProductContext(
        product_id=pid,
        product_name=pname,
        domain=domain,
        context_text=context_text,
        spec_summary=spec_summary,
        review_summary=review_summary,
        review_count=len(review_texts),
        metadata=metadata,
    )
```

- [ ] **Step 2.2: Write `tests/test_context.py`**

```python
# implementations/design_12_distilled_llm/tests/test_context.py
"""Tests for context.py — ProductContext construction with mocked LLM."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import (
    ProductContext,
    build_product_context,
    build_spec_summary,
)


def test_build_spec_summary_formats_metadata():
    metadata = {"length_cm": 177, "waist_width_mm": 100, "weight_g": 1850}
    result = build_spec_summary(metadata)
    assert "Length Cm: 177" in result
    assert "Waist Width Mm: 100" in result
    assert "Weight G: 1850" in result


def test_build_spec_summary_empty():
    assert build_spec_summary({}) == "No specs available."


def test_build_product_context_with_reviews():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Reviewers praise edge grip and stiffness."

    product = {
        "id": "SKI-001",
        "name": "Test Ski",
        "specs": {"length_cm": 170, "waist_width_mm": 66},
    }
    reviews = [
        {"product_id": "SKI-001", "text": "Great edge grip."},
        {"product_id": "SKI-001", "text": "Very stiff ski."},
    ]

    ctx = build_product_context(product, reviews, mock_llm, "ski")

    assert isinstance(ctx, ProductContext)
    assert ctx.product_id == "SKI-001"
    assert ctx.product_name == "Test Ski"
    assert ctx.domain == "ski"
    assert ctx.review_count == 2
    assert "Length Cm: 170" in ctx.spec_summary
    assert "Reviewers praise" in ctx.review_summary
    assert "Specs:" in ctx.context_text
    assert "Review consensus (2 reviews):" in ctx.context_text
    mock_llm.generate.assert_called_once()


def test_build_product_context_no_reviews():
    mock_llm = MagicMock()

    product = {"product_id": "SKI-002", "product_name": "Bare Ski", "metadata": {}}
    ctx = build_product_context(product, [], mock_llm, "ski")

    assert ctx.review_count == 0
    assert ctx.review_summary == "No reviews available."
    mock_llm.generate.assert_not_called()


def test_build_product_context_uses_alternate_keys():
    """Verify the builder handles both 'id'/'name' and 'product_id'/'product_name'."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Summary."

    product = {"product_id": "X-1", "product_name": "Alt Ski", "metadata": {"flex": 5}}
    reviews = [{"review_text": "Nice flex."}]

    ctx = build_product_context(product, reviews, mock_llm, "ski")
    assert ctx.product_id == "X-1"
    assert ctx.product_name == "Alt Ski"
```

- [ ] **Step 2.3: Run `pytest implementations/design_12_distilled_llm/tests/test_context.py -v` — verify all pass**

---

### Task 3: Teacher Labeling (`teacher.py`)

**Files:**
- Create: `implementations/design_12_distilled_llm/teacher.py`
- Create: `implementations/design_12_distilled_llm/tests/test_teacher.py`

- [ ] **Step 3.1: Create `teacher.py` with prompt template, `label_pair()`, and `label_all_pairs()`**

```python
# implementations/design_12_distilled_llm/teacher.py
"""Teacher labeling pipeline — prompt, scoring, caching."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_provider import LLMProvider

from .context import ProductContext
from .db import get_cached_judgment, insert_teacher_judgment

logger = logging.getLogger(__name__)


@dataclass
class TeacherJudgment:
    """A single teacher evaluation of a (query, product) pair."""

    query: str
    product_id: str
    product_name: str
    score: float
    explanation: str
    matched_attributes: dict[str, float]
    teacher_model: str
    timestamp: str


TEACHER_PROMPT = """\
You are an expert {domain} product recommender. A user has described what they \
want, and you need to evaluate how well a specific product matches their preferences.

User's query: "{query}"

Product: {product_name}
{product_context}

Evaluate this product against the user's query. Consider:
- How well each mentioned attribute matches the user's stated preferences
- Trade-offs: where the product excels vs. falls short
- Overall suitability, accounting for how important each attribute is to the query

Respond with ONLY valid JSON in this exact format:
{{
  "score": <float 0.0 to 1.0, where 1.0 is a perfect match>,
  "explanation": "<2-4 sentences explaining why this score, citing specific product characteristics>",
  "matched_attributes": {{
    "<attribute_name>": <float 0.0 to 1.0 indicating match strength>,
    ...
  }}
}}

Scoring guidelines:
- 0.9-1.0: Near-perfect match on all queried attributes
- 0.7-0.89: Strong match with minor gaps
- 0.5-0.69: Decent match but notable trade-offs
- 0.3-0.49: Partial match, significant misalignments
- 0.0-0.29: Poor match, wrong category or contradicts preferences"""


def label_pair(
    query: str,
    product_ctx: ProductContext,
    llm: LLMProvider,
    domain: str,
) -> TeacherJudgment:
    """Score a single (query, product) pair using the teacher model.

    Args:
        query: Natural language user query.
        product_ctx: Pre-built product context.
        llm: LLM provider instance (teacher model).
        domain: Product domain (e.g. "ski").

    Returns:
        A TeacherJudgment with score, explanation, and matched attributes.

    Raises:
        json.JSONDecodeError: If the teacher output is not valid JSON.
        KeyError: If required fields are missing from teacher output.
    """
    prompt = TEACHER_PROMPT.format(
        domain=domain,
        query=query,
        product_name=product_ctx.product_name,
        product_context=product_ctx.context_text,
    )
    response = llm.generate(prompt, json_mode=True)
    parsed = json.loads(response)

    score = max(0.0, min(1.0, float(parsed["score"])))

    return TeacherJudgment(
        query=query,
        product_id=product_ctx.product_id,
        product_name=product_ctx.product_name,
        score=score,
        explanation=parsed["explanation"],
        matched_attributes={
            k: max(0.0, min(1.0, float(v)))
            for k, v in parsed.get("matched_attributes", {}).items()
        },
        teacher_model=llm.llm_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def label_all_pairs(
    queries: list[str],
    products: list[ProductContext],
    llm: LLMProvider,
    domain: str,
    db: sqlite3.Connection,
) -> list[TeacherJudgment]:
    """Label all (query, product) pairs, skipping cached results.

    Results are persisted to SQLite as they are produced, so the process
    can be resumed after interruption.

    Args:
        queries: List of natural language queries.
        products: List of ProductContext objects.
        llm: LLM provider instance.
        domain: Product domain.
        db: Open SQLite connection with schema initialized.

    Returns:
        List of newly-created TeacherJudgment objects (excludes cached).
    """
    judgments: list[TeacherJudgment] = []
    total = len(queries) * len(products)
    completed = 0

    for query in queries:
        for product in products:
            cached = get_cached_judgment(db, query, product.product_id)
            if cached:
                completed += 1
                continue

            judgment = label_pair(query, product, llm, domain)

            insert_teacher_judgment(
                db,
                query=query,
                product_id=judgment.product_id,
                score=judgment.score,
                explanation=judgment.explanation,
                matched_attributes=judgment.matched_attributes,
                teacher_model=judgment.teacher_model,
                created_at=judgment.timestamp,
            )
            db.commit()

            judgments.append(judgment)
            completed += 1

            if completed % 50 == 0:
                logger.info("Labeled %d/%d pairs", completed, total)

    logger.info("Labeling complete: %d new, %d total", len(judgments), total)
    return judgments
```

- [ ] **Step 3.2: Write `tests/test_teacher.py`**

```python
# implementations/design_12_distilled_llm/tests/test_teacher.py
"""Tests for teacher.py — prompt formatting, JSON parsing, caching."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import ProductContext
from implementations.design_12_distilled_llm.db import init_db
from implementations.design_12_distilled_llm.teacher import (
    TEACHER_PROMPT,
    TeacherJudgment,
    label_all_pairs,
    label_pair,
)


def _make_ctx(pid: str = "SKI-001", name: str = "Test Ski") -> ProductContext:
    return ProductContext(
        product_id=pid,
        product_name=name,
        domain="ski",
        context_text="Specs: Length: 170cm\n\nReview consensus (3 reviews): Great ski.",
        spec_summary="Length: 170cm",
        review_summary="Great ski.",
        review_count=3,
    )


def _mock_llm(response_dict: dict) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(response_dict)
    llm.llm_model = "mock-teacher"
    return llm


def test_teacher_prompt_contains_placeholders():
    formatted = TEACHER_PROMPT.format(
        domain="ski",
        query="stiff carving ski",
        product_name="Test Ski",
        product_context="Specs: Length: 170cm",
    )
    assert "stiff carving ski" in formatted
    assert "Test Ski" in formatted
    assert "Specs: Length: 170cm" in formatted
    assert "expert ski product recommender" in formatted


def test_label_pair_parses_teacher_response():
    response = {
        "score": 0.85,
        "explanation": "Strong match on stiffness and edge hold.",
        "matched_attributes": {"stiffness": 0.9, "edge_grip": 0.8},
    }
    llm = _mock_llm(response)
    ctx = _make_ctx()

    judgment = label_pair("stiff carving ski", ctx, llm, "ski")

    assert isinstance(judgment, TeacherJudgment)
    assert judgment.score == 0.85
    assert judgment.explanation == "Strong match on stiffness and edge hold."
    assert judgment.matched_attributes["stiffness"] == 0.9
    assert judgment.teacher_model == "mock-teacher"
    assert judgment.product_id == "SKI-001"


def test_label_pair_clamps_score():
    llm = _mock_llm({"score": 1.5, "explanation": "x", "matched_attributes": {}})
    judgment = label_pair("q", _make_ctx(), llm, "ski")
    assert judgment.score == 1.0

    llm2 = _mock_llm({"score": -0.2, "explanation": "x", "matched_attributes": {}})
    judgment2 = label_pair("q", _make_ctx(), llm2, "ski")
    assert judgment2.score == 0.0


def test_label_all_pairs_caches_and_skips():
    response = {
        "score": 0.7,
        "explanation": "Decent match.",
        "matched_attributes": {"flex": 0.6},
    }
    llm = _mock_llm(response)
    db = init_db(":memory:")
    ctx = _make_ctx()

    # First run: labels the pair
    results = label_all_pairs(["query1"], [ctx], llm, "ski", db)
    assert len(results) == 1
    assert llm.generate.call_count == 1

    # Second run: skips due to cache
    results2 = label_all_pairs(["query1"], [ctx], llm, "ski", db)
    assert len(results2) == 0
    assert llm.generate.call_count == 1  # no new calls

    db.close()


def test_label_all_pairs_multiple_queries_and_products():
    response = {
        "score": 0.5,
        "explanation": "Partial match.",
        "matched_attributes": {},
    }
    llm = _mock_llm(response)
    db = init_db(":memory:")
    products = [_make_ctx("SKI-001", "Ski A"), _make_ctx("SKI-002", "Ski B")]

    results = label_all_pairs(["q1", "q2"], products, llm, "ski", db)
    assert len(results) == 4  # 2 queries x 2 products
    assert llm.generate.call_count == 4

    db.close()
```

- [ ] **Step 3.3: Run `pytest implementations/design_12_distilled_llm/tests/test_teacher.py -v` — verify all pass**

---

### Task 4: Dataset Preparation (`dataset.py`)

**Files:**
- Create: `implementations/design_12_distilled_llm/dataset.py`
- Create: `implementations/design_12_distilled_llm/tests/test_dataset.py`

- [ ] **Step 4.1: Create `dataset.py` with `TrainingExample`, `build_training_dataset()`, and `save_dataset_jsonl()`**

```python
# implementations/design_12_distilled_llm/dataset.py
"""Convert teacher judgments to instruction-tuning format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .context import ProductContext
from .teacher import TeacherJudgment


STUDENT_INSTRUCTION = (
    "You are a product recommendation assistant. Given a user query and a "
    "product description, evaluate how well the product matches the query. "
    "Respond with valid JSON containing score, explanation, and "
    "matched_attributes."
)


@dataclass
class TrainingExample:
    """A single instruction-tuning example for the student model."""

    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


def build_training_dataset(
    judgments: list[TeacherJudgment],
    contexts: dict[str, ProductContext],
) -> list[TrainingExample]:
    """Convert teacher judgments to instruction-tuning examples.

    Args:
        judgments: List of TeacherJudgment from the labeling pipeline.
        contexts: Dict mapping product_id -> ProductContext.

    Returns:
        List of TrainingExample ready for JSONL export.
    """
    dataset: list[TrainingExample] = []
    for j in judgments:
        ctx = contexts[j.product_id]
        input_text = (
            f"Rate this product for the query: {j.query}\n\n"
            f"Product: {ctx.product_name}\n"
            f"{ctx.context_text}"
        )
        output_text = json.dumps(
            {
                "score": round(j.score, 2),
                "explanation": j.explanation,
                "matched_attributes": {
                    k: round(v, 2) for k, v in j.matched_attributes.items()
                },
            },
            indent=2,
        )
        dataset.append(
            TrainingExample(
                instruction=STUDENT_INSTRUCTION,
                input=input_text,
                output=output_text,
            )
        )
    return dataset


def save_dataset_jsonl(
    examples: list[TrainingExample],
    output_path: str | Path,
) -> Path:
    """Write training examples to a JSONL file.

    Args:
        examples: List of TrainingExample objects.
        output_path: File path to write.

    Returns:
        The resolved Path of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    return path
```

- [ ] **Step 4.2: Write `tests/test_dataset.py`**

```python
# implementations/design_12_distilled_llm/tests/test_dataset.py
"""Tests for dataset.py — training data format and JSONL output."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import ProductContext
from implementations.design_12_distilled_llm.dataset import (
    STUDENT_INSTRUCTION,
    TrainingExample,
    build_training_dataset,
    save_dataset_jsonl,
)
from implementations.design_12_distilled_llm.teacher import TeacherJudgment


def _make_judgment() -> TeacherJudgment:
    return TeacherJudgment(
        query="stiff carving ski",
        product_id="SKI-001",
        product_name="Test Ski",
        score=0.85,
        explanation="Strong match on stiffness.",
        matched_attributes={"stiffness": 0.9, "edge_grip": 0.8},
        teacher_model="mock-teacher",
        timestamp="2026-01-01T00:00:00",
    )


def _make_context() -> ProductContext:
    return ProductContext(
        product_id="SKI-001",
        product_name="Test Ski",
        domain="ski",
        context_text="Specs: Length: 170cm\n\nReview consensus (3 reviews): Great ski.",
        spec_summary="Length: 170cm",
        review_summary="Great ski.",
        review_count=3,
    )


def test_build_training_dataset_format():
    judgments = [_make_judgment()]
    contexts = {"SKI-001": _make_context()}

    dataset = build_training_dataset(judgments, contexts)

    assert len(dataset) == 1
    ex = dataset[0]
    assert isinstance(ex, TrainingExample)
    assert ex.instruction == STUDENT_INSTRUCTION
    assert "stiff carving ski" in ex.input
    assert "Test Ski" in ex.input
    assert "Specs: Length: 170cm" in ex.input

    output_parsed = json.loads(ex.output)
    assert output_parsed["score"] == 0.85
    assert output_parsed["explanation"] == "Strong match on stiffness."
    assert output_parsed["matched_attributes"]["stiffness"] == 0.9


def test_training_example_to_dict():
    ex = TrainingExample(
        instruction="Inst",
        input="In",
        output="Out",
    )
    d = ex.to_dict()
    assert d == {"instruction": "Inst", "input": "In", "output": "Out"}


def test_save_dataset_jsonl():
    judgments = [_make_judgment()]
    contexts = {"SKI-001": _make_context()}
    dataset = build_training_dataset(judgments, contexts)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_dataset_jsonl(dataset, Path(tmp) / "train.jsonl")
        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert "instruction" in record
        assert "input" in record
        assert "output" in record
        assert "stiff carving ski" in record["input"]


def test_save_dataset_jsonl_multiple_examples():
    j1 = _make_judgment()
    j2 = TeacherJudgment(
        query="powder ski",
        product_id="SKI-001",
        product_name="Test Ski",
        score=0.3,
        explanation="Poor match for powder.",
        matched_attributes={"powder_float": 0.1},
        teacher_model="mock-teacher",
        timestamp="2026-01-01T00:00:00",
    )
    contexts = {"SKI-001": _make_context()}
    dataset = build_training_dataset([j1, j2], contexts)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_dataset_jsonl(dataset, Path(tmp) / "train.jsonl")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert "stiff carving ski" in r1["input"]
        assert "powder ski" in r2["input"]
```

- [ ] **Step 4.3: Run `pytest implementations/design_12_distilled_llm/tests/test_dataset.py -v` — verify all pass**

---

### Task 5: Student Training (`student.py`)

**Files:**
- Create: `implementations/design_12_distilled_llm/student.py`

- [ ] **Step 5.1: Create `student.py` with `train_student()`, LoRA config, and export helpers**

```python
# implementations/design_12_distilled_llm/student.py
"""LoRA training via unsloth and export options."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .db import insert_training_run

logger = logging.getLogger(__name__)

# Default LoRA target modules for Qwen 2.5
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _build_lora_config(rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> dict:
    """Return a dict of LoRA hyperparameters (for inspection/testing).

    Args:
        rank: LoRA rank.
        alpha: LoRA alpha (scaling factor).
        dropout: LoRA dropout rate.

    Returns:
        Dict with all LoRA config fields.
    """
    return {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": list(TARGET_MODULES),
        "lora_dropout": dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def _build_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
) -> dict:
    """Return a dict of TrainingArguments fields (for inspection/testing).

    Args:
        output_dir: Directory to save checkpoints and final adapter.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.

    Returns:
        Dict with all TrainingArguments fields.
    """
    return {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": True,
        "optim": "adamw_8bit",
    }


def _format_example(example: dict) -> dict:
    """Format a dataset row as an instruction-tuning prompt string.

    Args:
        example: Dict with "instruction", "input", "output" keys.

    Returns:
        Dict with a single "text" key containing the formatted prompt.
    """
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": prompt}


def train_student(
    dataset_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "adapters/design-12",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    db: sqlite3.Connection | None = None,
) -> str:
    """Fine-tune Qwen 2.5 on teacher judgments via LoRA.

    Requires: unsloth, peft, transformers, trl, datasets, torch.
    These are heavy dependencies and only imported inside this function.

    Args:
        dataset_path: Path to JSONL training data.
        base_model: HuggingFace model identifier.
        output_dir: Where to save the LoRA adapter.
        num_epochs: Training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        max_seq_length: Maximum sequence length.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        db: Optional SQLite connection to record the training run.

    Returns:
        The output_dir path where the adapter was saved.
    """
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    logger.info(
        "Starting training: base=%s, dataset=%s, epochs=%d, rank=%d",
        base_model, dataset_path, num_epochs, lora_rank,
    )

    # Load base model with unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )

    # Load and format dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(_format_example)

    num_examples = len(dataset)

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            **_build_training_args(output_dir, num_epochs, batch_size, learning_rate)
        ),
    )

    train_result = trainer.train()
    final_loss = train_result.training_loss

    # Save adapter
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete. Adapter saved to %s. Final loss: %.4f", output_dir, final_loss)

    # Record training run
    if db is not None:
        insert_training_run(
            db,
            base_model=base_model,
            lora_rank=lora_rank,
            num_examples=num_examples,
            num_epochs=num_epochs,
            final_loss=final_loss,
            adapter_path=output_dir,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.commit()

    return output_dir
```

- [ ] **Step 5.2: Verify config construction by running a quick inline check (no GPU needed)**

Run from project root:

```bash
cd /Users/harvey/Development/sports/recommend && python -c "
from implementations.design_12_distilled_llm.student import (
    _build_lora_config, _build_training_args, _format_example
)
import json

lora = _build_lora_config()
assert lora['r'] == 16
assert lora['task_type'] == 'CAUSAL_LM'
assert 'q_proj' in lora['target_modules']

args = _build_training_args('/tmp/test')
assert args['num_train_epochs'] == 3
assert args['optim'] == 'adamw_8bit'

ex = _format_example({'instruction': 'I', 'input': 'In', 'output': 'Out'})
assert '### Instruction:\nI' in ex['text']
assert '### Response:\nOut' in ex['text']

print('All student config checks passed.')
"
```

---

### Task 6: Inference (`inference.py`)

**Files:**
- Create: `implementations/design_12_distilled_llm/inference.py`
- Create: `implementations/design_12_distilled_llm/tests/test_inference.py`

- [ ] **Step 6.1: Create `inference.py` with `StudentJudgment`, `StudentInference` class**

```python
# implementations/design_12_distilled_llm/inference.py
"""Student model inference + JSON parsing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .context import ProductContext
from .dataset import STUDENT_INSTRUCTION

logger = logging.getLogger(__name__)


@dataclass
class StudentJudgment:
    """Student model's evaluation of a (query, product) pair."""

    product_id: str
    product_name: str
    score: float
    explanation: str
    matched_attributes: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""
    parse_success: bool = True


def _extract_json(text: str) -> str:
    """Extract the first JSON object from model output.

    Handles common issues: markdown fences, leading text, nested braces.

    Args:
        text: Raw model output string.

    Returns:
        The extracted JSON string.

    Raises:
        json.JSONDecodeError: If no valid JSON object is found.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # Find first { ... } block with brace matching
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise json.JSONDecodeError("Unterminated JSON object", text, start)


def _parse_judgment(
    raw_output: str,
    product_id: str,
    product_name: str,
) -> StudentJudgment:
    """Parse raw model output into a StudentJudgment.

    On parse failure, returns a fallback judgment with score=0.0.

    Args:
        raw_output: Raw text from the student model.
        product_id: The product being evaluated.
        product_name: The product name.

    Returns:
        A StudentJudgment, with parse_success=False on failure.
    """
    try:
        json_str = _extract_json(raw_output)
        parsed = json.loads(json_str)
        return StudentJudgment(
            product_id=product_id,
            product_name=product_name,
            score=max(0.0, min(1.0, float(parsed["score"]))),
            explanation=parsed.get("explanation", ""),
            matched_attributes={
                k: max(0.0, min(1.0, float(v)))
                for k, v in parsed.get("matched_attributes", {}).items()
            },
            raw_output=raw_output,
            parse_success=True,
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(
            "Failed to parse student output for %s: %s", product_name, e
        )
        return StudentJudgment(
            product_id=product_id,
            product_name=product_name,
            score=0.0,
            explanation=f"Parse error: {e}",
            matched_attributes={},
            raw_output=raw_output,
            parse_success=False,
        )


class StudentInference:
    """Wraps a loaded student model for inference.

    The model and tokenizer are injected so this class is testable
    without actually loading a transformer model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_seq_length: int = 2048,
    ):
        """Initialize with a loaded model and tokenizer.

        Args:
            model: A HuggingFace model (base + LoRA merged or PeftModel).
            tokenizer: The corresponding tokenizer.
            max_seq_length: Maximum input length for truncation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def _build_prompt(self, query: str, product_ctx: ProductContext) -> str:
        """Build the inference prompt for a (query, product) pair."""
        return (
            f"### Instruction:\n{STUDENT_INSTRUCTION}\n\n"
            f"### Input:\n"
            f"Rate this product for the query: {query}\n\n"
            f"Product: {product_ctx.product_name}\n"
            f"{product_ctx.context_text}\n\n"
            f"### Response:\n"
        )

    def infer(self, query: str, product_ctx: ProductContext) -> StudentJudgment:
        """Run inference for a single (query, product) pair.

        Args:
            query: Natural language user query.
            product_ctx: Pre-built product context.

        Returns:
            A StudentJudgment with the model's evaluation.
        """
        import torch

        prompt = self._build_prompt(query, product_ctx)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)

        return _parse_judgment(raw_output, product_ctx.product_id, product_ctx.product_name)
```

- [ ] **Step 6.2: Write `tests/test_inference.py`**

```python
# implementations/design_12_distilled_llm/tests/test_inference.py
"""Tests for inference.py — JSON extraction, parsing, fallback behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from implementations.design_12_distilled_llm.inference import (
    StudentJudgment,
    _extract_json,
    _parse_judgment,
)


class TestExtractJson:
    """Test _extract_json with various model output formats."""

    def test_clean_json(self):
        text = '{"score": 0.8, "explanation": "Good.", "matched_attributes": {}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.8

    def test_markdown_fenced_json(self):
        text = '```json\n{"score": 0.7, "explanation": "OK.", "matched_attributes": {}}\n```'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.7

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"score": 0.6, "explanation": "Fair.", "matched_attributes": {}}\n```'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.6

    def test_leading_text(self):
        text = 'Here is my evaluation:\n{"score": 0.5, "explanation": "Partial.", "matched_attributes": {}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.5

    def test_nested_braces(self):
        text = '{"score": 0.9, "explanation": "Great.", "matched_attributes": {"stiffness": 0.9, "grip": 0.8}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["matched_attributes"]["stiffness"] == 0.9

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError, match="No JSON object found"):
            _extract_json("No JSON here at all")

    def test_unterminated_json_raises(self):
        with pytest.raises(json.JSONDecodeError, match="Unterminated"):
            _extract_json('{"score": 0.5, "explanation": "cut off')

    def test_trailing_text(self):
        text = '{"score": 0.4, "explanation": "Low.", "matched_attributes": {}}\nSome extra text.'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.4


class TestParseJudgment:
    """Test _parse_judgment with valid and invalid outputs."""

    def test_valid_output(self):
        raw = json.dumps({
            "score": 0.85,
            "explanation": "Strong match.",
            "matched_attributes": {"stiffness": 0.9},
        })
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.parse_success is True
        assert j.score == 0.85
        assert j.explanation == "Strong match."
        assert j.matched_attributes["stiffness"] == 0.9
        assert j.product_id == "SKI-001"

    def test_score_clamping(self):
        raw = json.dumps({"score": 2.0, "explanation": "x", "matched_attributes": {}})
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.score == 1.0

        raw2 = json.dumps({"score": -1.0, "explanation": "x", "matched_attributes": {}})
        j2 = _parse_judgment(raw2, "SKI-001", "Test Ski")
        assert j2.score == 0.0

    def test_malformed_json_fallback(self):
        j = _parse_judgment("not json at all", "SKI-001", "Test Ski")
        assert j.parse_success is False
        assert j.score == 0.0
        assert "Parse error" in j.explanation
        assert j.raw_output == "not json at all"

    def test_missing_score_key_fallback(self):
        raw = json.dumps({"explanation": "No score.", "matched_attributes": {}})
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.parse_success is False
        assert j.score == 0.0

    def test_missing_explanation_uses_empty(self):
        raw = json.dumps({"score": 0.5, "matched_attributes": {}})
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.parse_success is True
        assert j.explanation == ""

    def test_attribute_values_clamped(self):
        raw = json.dumps({
            "score": 0.5,
            "explanation": "x",
            "matched_attributes": {"a": 1.5, "b": -0.3},
        })
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.matched_attributes["a"] == 1.0
        assert j.matched_attributes["b"] == 0.0
```

- [ ] **Step 6.3: Run `pytest implementations/design_12_distilled_llm/tests/test_inference.py -v` — verify all pass**

---

### Task 7: Recommender (`recommender.py`)

**Files:**
- Create: `implementations/design_12_distilled_llm/recommender.py`
- Create: `implementations/design_12_distilled_llm/tests/test_recommender.py`

- [ ] **Step 7.1: Create `recommender.py` implementing the `Recommender` protocol**

```python
# implementations/design_12_distilled_llm/recommender.py
"""Design #12: Distilled LLM Ranker.

Uses a teacher LLM to label query-product pairs at training time,
then a LoRA-tuned Qwen 2.5 student for local inference at query time.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .context import ProductContext, build_product_context
from .db import init_db, upsert_product_context
from .inference import StudentInference, StudentJudgment, _parse_judgment

logger = logging.getLogger(__name__)


class DistilledLLMRecommender:
    """Implements the Recommender protocol using a distilled student model.

    Training-time: uses a teacher LLM to label (query, product) pairs.
    Query-time: uses a LoRA-tuned Qwen 2.5 student model locally.
    """

    def __init__(
        self,
        adapter_path: str = "adapters/design-12",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        db_path: str = "design_12.db",
        device: str = "auto",
        max_seq_length: int = 2048,
        llm: LLMProvider | None = None,
        inference_backend: StudentInference | None = None,
    ):
        """Initialize the recommender.

        Args:
            adapter_path: Path to the LoRA adapter directory.
            base_model: HuggingFace base model identifier.
            db_path: Path to SQLite database (":memory:" for in-memory).
            device: Device for model loading ("auto", "cpu", "cuda", "mps").
            max_seq_length: Maximum sequence length for inference.
            llm: Optional LLM provider override (default: from env).
            inference_backend: Optional pre-built StudentInference (for testing).
        """
        self.adapter_path = Path(adapter_path)
        self.base_model_name = base_model
        self.db = init_db(db_path)
        self.max_seq_length = max_seq_length
        self.device = device

        self.contexts: dict[str, ProductContext] = {}
        self.product_names: dict[str, str] = {}
        self._domain: str = ""

        self.llm = llm or get_provider()
        self._inference: StudentInference | None = inference_backend

    @property
    def inference(self) -> StudentInference:
        """Lazy-load the student model on first query."""
        if self._inference is None:
            self._inference = self._load_student()
        return self._inference

    def _load_student(self) -> StudentInference:
        """Load base model + LoRA adapter for inference."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading student model: %s + %s",
            self.base_model_name,
            self.adapter_path,
        )

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        if self.adapter_path.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, str(self.adapter_path))
        else:
            logger.warning(
                "Adapter not found at %s; using base model (untrained). "
                "Run the training pipeline first.",
                self.adapter_path,
            )
            model = base

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        return StudentInference(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
        )

    def ingest(
        self, products: list[dict], reviews: list[dict], domain: str
    ) -> None:
        """Build product contexts from reviews + specs and store them.

        Args:
            products: List of product dicts (keys: id/product_id, name/product_name,
                      specs/metadata).
            reviews: List of review dicts (keys: product_id, text/review_text).
            domain: Product domain (e.g. "ski").
        """
        self._domain = domain
        reviews_by_product: dict[str, list[dict]] = defaultdict(list)
        for r in reviews:
            pid = r.get("product_id", r.get("id", ""))
            reviews_by_product[pid].append(r)

        for product in products:
            pid = product.get("product_id", product.get("id", ""))
            product_reviews = reviews_by_product.get(pid, [])

            ctx = build_product_context(product, product_reviews, self.llm, domain)
            self.contexts[pid] = ctx
            self.product_names[pid] = ctx.product_name

            upsert_product_context(
                self.db,
                product_id=ctx.product_id,
                product_name=ctx.product_name,
                domain=domain,
                context_text=ctx.context_text,
                spec_summary=ctx.spec_summary,
                review_summary=ctx.review_summary,
                review_count=ctx.review_count,
                metadata=ctx.metadata,
                built_at=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            )

        self.db.commit()
        logger.info(
            "Ingested %d products for domain '%s'", len(products), domain
        )

    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Score all products using the distilled student model.

        Args:
            query_text: Natural language query.
            domain: Product domain to search in.
            top_k: Number of results to return.

        Returns:
            List of RecommendationResult sorted by score descending.
        """
        candidates = [
            ctx
            for ctx in self.contexts.values()
            if ctx.domain == domain or domain == ""
        ]

        if not candidates:
            logger.warning("No products found for domain '%s'", domain)
            return []

        scored: list[StudentJudgment] = []
        for ctx in candidates:
            judgment = self.inference.infer(query_text, ctx)
            scored.append(judgment)

        scored.sort(key=lambda j: j.score, reverse=True)

        results: list[RecommendationResult] = []
        for j in scored[:top_k]:
            results.append(
                RecommendationResult(
                    product_id=j.product_id,
                    product_name=j.product_name,
                    score=round(j.score, 4),
                    explanation=j.explanation,
                    matched_attributes=j.matched_attributes,
                )
            )
        return results
```

- [ ] **Step 7.2: Write `tests/test_recommender.py` — integration test with mocked inference**

```python
# implementations/design_12_distilled_llm/tests/test_recommender.py
"""Integration tests for recommender.py with mocked student inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import ProductContext
from implementations.design_12_distilled_llm.inference import (
    StudentInference,
    StudentJudgment,
)
from implementations.design_12_distilled_llm.recommender import (
    DistilledLLMRecommender,
)
from shared.interface import RecommendationResult


def _make_mock_inference() -> MagicMock:
    """Create a mock StudentInference that returns predictable judgments."""
    mock = MagicMock(spec=StudentInference)

    def fake_infer(query: str, ctx: ProductContext) -> StudentJudgment:
        # Higher score for products with "carving" in the name when query mentions carving
        score = 0.8 if "carving" in ctx.product_name.lower() else 0.3
        return StudentJudgment(
            product_id=ctx.product_id,
            product_name=ctx.product_name,
            score=score,
            explanation=f"Evaluated {ctx.product_name} for: {query}",
            matched_attributes={"stiffness": score},
            raw_output="mock",
            parse_success=True,
        )

    mock.infer.side_effect = fake_infer
    return mock


def _make_mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = "Reviewers praise this ski."
    llm.llm_model = "mock-teacher"
    return llm


def test_recommender_ingest_and_query():
    mock_llm = _make_mock_llm()
    mock_inference = _make_mock_inference()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=mock_inference,
    )

    products = [
        {"id": "SKI-001", "name": "Expert Carving Ski", "specs": {"length_cm": 170}},
        {"id": "SKI-002", "name": "Powder Floater", "specs": {"length_cm": 185}},
    ]
    reviews = [
        {"product_id": "SKI-001", "text": "Great edge grip."},
        {"product_id": "SKI-002", "text": "Floats in powder."},
    ]

    rec.ingest(products, reviews, "ski")

    assert len(rec.contexts) == 2
    assert "SKI-001" in rec.contexts
    assert "SKI-002" in rec.contexts

    results = rec.query("stiff carving ski", "ski", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RecommendationResult) for r in results)
    # Carving ski should rank first
    assert results[0].product_id == "SKI-001"
    assert results[0].score > results[1].score
    assert 0.0 <= results[0].score <= 1.0
    assert len(results[0].explanation) > 0
    assert isinstance(results[0].matched_attributes, dict)


def test_recommender_query_empty_domain():
    mock_llm = _make_mock_llm()
    mock_inference = _make_mock_inference()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=mock_inference,
    )

    results = rec.query("anything", "ski", top_k=5)
    assert results == []


def test_recommender_top_k_limits_results():
    mock_llm = _make_mock_llm()
    mock_inference = _make_mock_inference()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=mock_inference,
    )

    products = [
        {"id": f"SKI-{i:03d}", "name": f"Ski {i}", "specs": {}}
        for i in range(10)
    ]
    rec.ingest(products, [], "ski")

    results = rec.query("test", "ski", top_k=3)
    assert len(results) == 3


def test_recommender_stores_contexts_in_db():
    mock_llm = _make_mock_llm()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=_make_mock_inference(),
    )

    products = [{"id": "SKI-001", "name": "Test Ski", "specs": {"flex": 7}}]
    rec.ingest(products, [], "ski")

    row = rec.db.execute(
        "SELECT * FROM product_contexts WHERE product_id = 'SKI-001'"
    ).fetchone()
    assert row is not None
    assert row["product_name"] == "Test Ski"
    assert row["domain"] == "ski"
```

- [ ] **Step 7.3: Run `pytest implementations/design_12_distilled_llm/tests/test_recommender.py -v` — verify all pass**

---

### Task 8: Package Setup (`__init__.py`, `requirements.txt`)

**Files:**
- Create: `implementations/design_12_distilled_llm/__init__.py`
- Create: `implementations/design_12_distilled_llm/requirements.txt`

- [ ] **Step 8.1: Create `__init__.py` with `create_recommender()` factory**

```python
# implementations/design_12_distilled_llm/__init__.py
"""Design #12: Distilled LLM Ranker Recommendation System."""

from .recommender import DistilledLLMRecommender


def create_recommender() -> DistilledLLMRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return DistilledLLMRecommender()
```

- [ ] **Step 8.2: Create `requirements.txt`**

```
# implementations/design_12_distilled_llm/requirements.txt
unsloth
peft
transformers
trl
datasets
torch
httpx
pytest
```

- [ ] **Step 8.3: Verify import works**

Run from project root:

```bash
cd /Users/harvey/Development/sports/recommend && python -c "
from implementations.design_12_distilled_llm import create_recommender
print(f'Factory function: {create_recommender}')
print(f'Return type annotation: {create_recommender.__annotations__}')
print('Import OK.')
"
```

---

### Task 9: Benchmark Smoke Test

**Files:** None created — verification only.

- [ ] **Step 9.1: Verify the recommender loads, ingests mock data, and returns valid `RecommendationResult` objects**

Run from project root:

```bash
cd /Users/harvey/Development/sports/recommend && python -c "
from unittest.mock import MagicMock
from implementations.design_12_distilled_llm.recommender import DistilledLLMRecommender
from implementations.design_12_distilled_llm.inference import StudentInference, StudentJudgment
from shared.interface import Recommender, RecommendationResult

# Verify protocol compliance
assert isinstance(DistilledLLMRecommender, type)

# Create with mocks (no GPU needed)
mock_llm = MagicMock()
mock_llm.generate.return_value = 'Reviewers say it is great.'
mock_llm.llm_model = 'mock'

mock_inference = MagicMock(spec=StudentInference)
mock_inference.infer.return_value = StudentJudgment(
    product_id='SKI-001',
    product_name='Test Ski',
    score=0.75,
    explanation='Good match on stiffness.',
    matched_attributes={'stiffness': 0.8},
    raw_output='mock',
    parse_success=True,
)

rec = DistilledLLMRecommender(
    db_path=':memory:',
    llm=mock_llm,
    inference_backend=mock_inference,
)

# Ingest
rec.ingest(
    products=[{'id': 'SKI-001', 'name': 'Test Ski', 'specs': {'length_cm': 170}}],
    reviews=[{'product_id': 'SKI-001', 'text': 'Stiff and fast.'}],
    domain='ski',
)

# Query
results = rec.query('stiff carving ski', 'ski', top_k=5)
assert len(results) == 1
r = results[0]
assert isinstance(r, RecommendationResult)
assert r.product_id == 'SKI-001'
assert r.product_name == 'Test Ski'
assert 0.0 <= r.score <= 1.0
assert len(r.explanation) > 0
assert isinstance(r.matched_attributes, dict)

print('Smoke test PASSED: recommender ingests, queries, returns valid RecommendationResult.')
"
```

- [ ] **Step 9.2: Run the full test suite**

```bash
cd /Users/harvey/Development/sports/recommend && python -m pytest implementations/design_12_distilled_llm/tests/ -v
```

All tests should pass. If any fail, fix and re-run before proceeding.
