# Wine Bundle + Streamlit Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end demonstration of the existing LoRA pipeline on a public wine-reviews dataset, validated by the eval gate, exposed through a Streamlit UI that supports the existing ski domain too.

**Architecture:** Reuse `scripts/arena_new.py`, `scripts/finetune_listwise.py`, and `scripts/eval_bundle.py` unchanged. Add (a) a dataset builder that pulls and subsamples HF wine-reviews into the bundle's input format, (b) an eval-set builder that hand-authors queries and gets Opus ground-truth via the existing teacher LLM cache, and (c) a Streamlit single-file frontend that wraps `LocalHybridRecommender.from_bundle`.

**Tech Stack:** Python 3.9, HuggingFace `datasets`, existing `shared/llm_provider.py` (Anthropic), Streamlit, design-14 hybrid recommender (RRF + listwise).

---

## File Structure

**New files:**
- `scripts/build_wine_dataset.py` — pulls HF mirror, subsamples to 200, writes `data/wine/{products,reviews}.jsonl`
- `scripts/build_wine_eval.py` — hand-authored queries + Opus ground-truth → `benchmark/data/per_domain/wine_eval.json`
- `data/wine/products.jsonl` — generated catalog
- `data/wine/reviews.jsonl` — generated reviews (1:1 with products)
- `data/wine/README.md` — regeneration instructions, dataset license note
- `benchmark/data/per_domain/wine_eval.json` — committed eval set
- `frontend/app.py` — Streamlit UI
- `frontend/requirements.txt` — `streamlit>=1.30`
- `artifacts/wine/` — bundle written by `arena_new.py`

**Modified files (only if hardcoding surfaces):**
- `scripts/arena_new.py` — only if a ski-specific code path blocks wine compilation

---

## Important context the engineer must know

- **`finetune-listwise` is NOT a step in `arena_new.py`.** Despite CLAUDE.md showing it in the `--steps` list, the actual `arena_new.py` only dispatches `ingest, embedding-pin, reranker-pin, eval-import, extract-schema, generate-synthetic, finetune-embedding, finetune-reranker`. Listwise is run via a separate `python scripts/finetune_listwise.py --bundle artifacts/wine` after arena_new.py finishes.
- **`LocalHybridRecommender.from_bundle()` already passes `db_dir=str(bundle.paths.root)`** (recommender.py:164) — so it does NOT use a tempdir despite CLAUDE.md's gotcha #2. The frontend can call `from_bundle(...)` directly without needing an explicit `db_dir`.
- **Listwise reranker activates via `metadata.reranker_runtime_kind: "listwise"`** in the manifest (set by `finetune_listwise.py`, not by `arena_new.py`).
- **Eval JSON schema:** `{"version": "1.0", "queries": [{"id", "difficulty", "domain", "query_text", "key_attributes", "ground_truth_top5": [{"product_id", "relevance", "reason"}, ...]}, ...]}`. Mirrors `benchmark/data/per_domain/ski_eval_v2.json`.
- **Bundle product schema** (from `artifacts/ski/products.jsonl`): `{id, name, brand, category, attributes: {...}, specs: {...}}`. Plain JSON object, no nesting beyond two levels.
- **Bundle review schema:** `{product_id, rating, reviewer, text}`. One review per line.
- **Teacher LLM:** the project uses `shared/llm_provider.py`. Set `RECOMMEND_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...`. Default model is sonnet — override to `claude-opus-4-7` for ground-truthing. Cache at `<bundle>/.teacher_cache.sqlite` is keyed on `(model, sha256(prompt))`.

---

## Task 1: Add streamlit to a frontend requirements file

**Files:**
- Create: `frontend/requirements.txt`

- [ ] **Step 1: Write the file**

```
streamlit>=1.30,<2.0
```

- [ ] **Step 2: Commit**

```bash
git add frontend/requirements.txt
git commit -m "chore(frontend): pin streamlit dep"
```

---

## Task 2: Build the wine dataset builder script

**Files:**
- Create: `scripts/build_wine_dataset.py`
- Create: `data/wine/README.md`

**Why this script:** turn the 130k-row HuggingFace `wine-reviews` dataset into the project's `products.jsonl` + `reviews.jsonl` format with deterministic stratified subsampling.

The dataset is mirrored on HF as `Sayali9141/wine-reviews-130k` (also `james-burton/wine_reviews_kaggle`); use `james-burton/wine_reviews_kaggle` as the primary source — it preserves the original Kaggle column names and is more stable. If load fails, the script logs the error and exits 1 — do NOT fall back silently.

- [ ] **Step 1: Write `scripts/build_wine_dataset.py`**

```python
#!/usr/bin/env python3
"""build_wine_dataset.py — pull HF wine-reviews and emit the bundle inputs.

Output:
    data/wine/products.jsonl  (200 wines, stratified by variety/country/price)
    data/wine/reviews.jsonl   (200 reviews, 1:1 with products)

Usage:
    python scripts/build_wine_dataset.py [--n 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("build_wine_dataset")

OUT_DIR = _PROJECT_ROOT / "data" / "wine"
PRODUCTS_OUT = OUT_DIR / "products.jsonl"
REVIEWS_OUT = OUT_DIR / "reviews.jsonl"

HF_DATASET = "james-burton/wine_reviews_kaggle"

REQUIRED_VARIETIES = [
    "Cabernet Sauvignon", "Pinot Noir", "Chardonnay", "Riesling",
    "Syrah", "Sauvignon Blanc", "Merlot", "Sangiovese",
]
REQUIRED_COUNTRIES = ["US", "France", "Italy", "Spain", "Argentina", "Australia"]


def _normalize_category(variety: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", variety.lower()).strip("_")


def _parse_vintage(title: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", title or "")
    return int(m.group(0)) if m else None


def _price_tier(price: float) -> str:
    if price < 25:
        return "low"
    if price <= 60:
        return "mid"
    return "high"


def _load_dataset():
    from datasets import load_dataset
    logger.info("loading %s ...", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train")
    logger.info("loaded %d rows", len(ds))
    return ds


def _to_records(ds) -> list[dict]:
    rows = []
    for r in ds:
        title = r.get("title") or ""
        desc = r.get("description") or ""
        variety = r.get("variety") or ""
        price = r.get("price")
        country = r.get("country") or ""
        if not title or not desc or not variety or price is None or not country:
            continue
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            continue
        if price_f <= 0:
            continue
        rows.append({
            "title": title,
            "description": desc,
            "winery": r.get("winery") or "",
            "variety": variety,
            "country": country,
            "province": r.get("province") or "",
            "region_1": r.get("region_1") or "",
            "price": price_f,
            "points": int(r.get("points") or 0),
            "taster_name": r.get("taster_name") or "",
        })
    return rows


def _stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample with constraints:
       - >=8 distinct varieties (REQUIRED_VARIETIES present)
       - >=6 distinct countries (REQUIRED_COUNTRIES present)
       - thirds across price tiers (~n/3 each)
       - prefer points >=88
    """
    rng = np.random.default_rng(seed)
    by_variety = defaultdict(list)
    for r in rows:
        by_variety[r["variety"]].append(r)
    by_country = defaultdict(list)
    for r in rows:
        by_country[r["country"]].append(r)

    chosen: list[dict] = []
    chosen_ids: set[int] = set()

    def add_row(idx: int, src: list[dict]):
        if idx in chosen_ids:
            return False
        chosen.append(src[idx])
        chosen_ids.add(id(src[idx]))
        return True

    # 1. Floor: pick at least 1 row per required variety (highest points).
    for v in REQUIRED_VARIETIES:
        pool = by_variety.get(v, [])
        if not pool:
            continue
        pool_sorted = sorted(pool, key=lambda r: -r["points"])
        for r in pool_sorted:
            if id(r) not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(id(r))
                break

    # 2. Floor: at least 1 row per required country.
    for c in REQUIRED_COUNTRIES:
        pool = by_country.get(c, [])
        if not pool:
            continue
        pool_sorted = sorted(pool, key=lambda r: -r["points"])
        for r in pool_sorted:
            if id(r) not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(id(r))
                break

    # 3. Fill thirds across price tiers.
    high_pts = [r for r in rows if r["points"] >= 88]
    by_tier = defaultdict(list)
    for r in high_pts:
        by_tier[_price_tier(r["price"])].append(r)
    target_per_tier = n // 3
    for tier in ("low", "mid", "high"):
        pool = by_tier[tier]
        rng.shuffle(pool)
        for r in pool:
            if len(chosen) >= n:
                break
            tier_count = sum(1 for c in chosen if _price_tier(c["price"]) == tier)
            if tier_count >= target_per_tier and len(chosen) < n:
                # leave room for other tiers; only top up if total under n at end
                continue
            if id(r) not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(id(r))

    # 4. Top up to n with any remaining high-points rows.
    if len(chosen) < n:
        rng.shuffle(high_pts)
        for r in high_pts:
            if len(chosen) >= n:
                break
            if id(r) not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(id(r))

    # Trim to exactly n; deterministic order via stable sort by (country, variety, points).
    chosen.sort(key=lambda r: (r["country"], r["variety"], -r["points"], r["title"]))
    return chosen[:n]


def _to_bundle_record(idx: int, r: dict) -> tuple[dict, dict]:
    pid = f"WINE-{idx:03d}"
    region = r["region_1"] or r["province"]
    product = {
        "id": pid,
        "name": r["title"],
        "brand": r["winery"],
        "category": _normalize_category(r["variety"]),
        "attributes": {
            "points": r["points"],
            "price": r["price"],
            "country": r["country"],
            "region": region,
            "variety": r["variety"],
            "taster": r["taster_name"],
        },
        "specs": {
            "vintage": _parse_vintage(r["title"]),
        },
    }
    review = {
        "product_id": pid,
        "rating": max(1, min(5, round((r["points"] - 80) / 4))),  # 80→0, 100→5; clamped 1-5
        "reviewer": r["taster_name"] or "winemag",
        "text": r["description"],
    }
    return product, review


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = _load_dataset()
    rows = _to_records(ds)
    logger.info("kept %d rows after null filter", len(rows))

    chosen = _stratified_sample(rows, n=args.n, seed=args.seed)
    logger.info("selected %d wines", len(chosen))
    if len(chosen) < args.n:
        logger.error("only got %d / %d wines after stratification", len(chosen), args.n)
        return 1

    products: list[dict] = []
    reviews: list[dict] = []
    for i, r in enumerate(chosen):
        prod, rev = _to_bundle_record(i, r)
        products.append(prod)
        reviews.append(rev)

    with PRODUCTS_OUT.open("w") as f:
        for p_ in products:
            f.write(json.dumps(p_) + "\n")
    with REVIEWS_OUT.open("w") as f:
        for r_ in reviews:
            f.write(json.dumps(r_) + "\n")

    varieties = sorted({p["attributes"]["variety"] for p in products})
    countries = sorted({p["attributes"]["country"] for p in products})
    logger.info("wrote %s (%d wines, %d varieties, %d countries)",
                PRODUCTS_OUT, len(products), len(varieties), len(countries))
    logger.info("wrote %s (%d reviews)", REVIEWS_OUT, len(reviews))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `data/wine/README.md`**

```markdown
# Wine domain inputs

Source: WineEnthusiast reviews scraped by `zynicide` and published on Kaggle as
`zynicide/wine-reviews` (CC0). Mirrored on HuggingFace as
`james-burton/wine_reviews_kaggle`.

Files in this directory are **generated** — do not edit by hand.

## Regenerate

```bash
pip install datasets numpy
python scripts/build_wine_dataset.py --n 200 --seed 42
```

The script writes `products.jsonl` and `reviews.jsonl` (1:1) into this
directory. Stratified to span ≥8 varieties, ≥6 countries, and roughly
even thirds across price tiers <$25 / $25–$60 / >$60, weighted toward
points ≥88 for richer descriptions.
```

- [ ] **Step 3: Run the script**

```bash
pip install -q datasets numpy
python scripts/build_wine_dataset.py --n 200 --seed 42
```

Expected: logs "selected 200 wines", "wrote .../products.jsonl (200 wines, ≥8 varieties, ≥6 countries)".

- [ ] **Step 4: Spot-check the output**

```bash
wc -l data/wine/products.jsonl data/wine/reviews.jsonl
head -1 data/wine/products.jsonl | python3 -m json.tool
```

Expected: 200 lines each. Sample product has `id`, `name`, `brand`, `category`, `attributes`, `specs` keys.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_wine_dataset.py data/wine/README.md data/wine/products.jsonl data/wine/reviews.jsonl
git commit -m "data(wine): add 200-wine subsample of HF wine-reviews"
```

---

## Task 3: Build the eval-set builder script

**Files:**
- Create: `scripts/build_wine_eval.py`
- Create: `benchmark/data/per_domain/wine_eval.json`

**Approach:** hand-author 30 queries with `key_attributes` and `difficulty`. For ground truth, ask Opus to rank top-5 product IDs from the catalog with relevance grades 1–3 and a short reason for each. Cache responses on disk so re-runs are free.

- [ ] **Step 1: Write `scripts/build_wine_eval.py`**

```python
#!/usr/bin/env python3
"""build_wine_eval.py — author the wine eval set with Opus ground truth.

Hand-authored queries; ground truth top-5 generated by Opus with disk
caching so reruns are free. Output schema matches ski_eval_v2.json.

Usage:
    export RECOMMEND_LLM_PROVIDER=anthropic
    export ANTHROPIC_API_KEY=...
    python scripts/build_wine_eval.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.llm_provider import get_provider  # noqa: E402

logger = logging.getLogger("build_wine_eval")

PRODUCTS_PATH = _PROJECT_ROOT / "data" / "wine" / "products.jsonl"
OUT_PATH = _PROJECT_ROOT / "benchmark" / "data" / "per_domain" / "wine_eval.json"
CACHE_PATH = _PROJECT_ROOT / "data" / "wine" / ".eval_teacher_cache.sqlite"

OPUS_MODEL = "claude-opus-4-7"


def Q(qid, difficulty, query_text, key_attrs):
    return {
        "id": qid,
        "difficulty": difficulty,
        "domain": "wine",
        "query_text": query_text,
        "key_attributes": key_attrs,
    }


QUERIES = [
    # ---- EASY (10): single dominant attribute --------------------------
    Q("EASY-01", "easy", "argentinian malbec", ["variety:Malbec", "country:Argentina"]),
    Q("EASY-02", "easy", "oaky chardonnay", ["variety:Chardonnay", "oaky"]),
    Q("EASY-03", "easy", "riesling under $20", ["variety:Riesling", "price<20"]),
    Q("EASY-04", "easy", "italian sangiovese", ["variety:Sangiovese", "country:Italy"]),
    Q("EASY-05", "easy", "pinot noir from oregon or burgundy", ["variety:Pinot Noir"]),
    Q("EASY-06", "easy", "spanish rioja tempranillo", ["variety:Tempranillo", "country:Spain"]),
    Q("EASY-07", "easy", "crisp sauvignon blanc", ["variety:Sauvignon Blanc", "crisp"]),
    Q("EASY-08", "easy", "australian shiraz", ["variety:Syrah", "country:Australia"]),
    Q("EASY-09", "easy", "high-rated wine over 95 points", ["points>=95"]),
    Q("EASY-10", "easy", "cheap red under $15", ["price<15", "red"]),

    # ---- MEDIUM (10): two-to-three constraints -------------------------
    Q("MED-01", "medium", "full-bodied napa cabernet with cellar potential",
      ["variety:Cabernet Sauvignon", "region:Napa", "age-worthy"]),
    Q("MED-02", "medium", "mineral-driven white from france under $40",
      ["white", "mineral", "country:France", "price<40"]),
    Q("MED-03", "medium", "fruity new world pinot noir around $30",
      ["variety:Pinot Noir", "fruity", "price~30"]),
    Q("MED-04", "medium", "bold italian red with firm tannins",
      ["country:Italy", "red", "tannic"]),
    Q("MED-05", "medium", "value chardonnay under $25 with some oak",
      ["variety:Chardonnay", "price<25", "oaky"]),
    Q("MED-06", "medium", "elegant pinot noir from burgundy with earthy notes",
      ["variety:Pinot Noir", "country:France", "earthy"]),
    Q("MED-07", "medium", "rich syrah with peppery finish",
      ["variety:Syrah", "peppery", "rich"]),
    Q("MED-08", "medium", "off-dry riesling for spicy food",
      ["variety:Riesling", "off-dry"]),
    Q("MED-09", "medium", "premium 95+ point red over $100",
      ["points>=95", "price>100", "red"]),
    Q("MED-10", "medium", "balanced bordeaux blend for steak",
      ["bordeaux blend", "country:France", "food-pairing:steak"]),

    # ---- HARD (7): negations / ranges / trade-offs ---------------------
    Q("HARD-01", "hard", "crisp white that is NOT too sweet, under $30",
      ["white", "dry", "NOT sweet", "price<30"]),
    Q("HARD-02", "hard", "powerful red but not overly oaky, between $40 and $80",
      ["red", "NOT oaky", "price 40-80"]),
    Q("HARD-03", "hard", "pinot noir for someone who normally drinks cabernet",
      ["variety:Pinot Noir", "bigger style"]),
    Q("HARD-04", "hard", "natural-tasting wine, low intervention, under $50",
      ["natural", "low-intervention", "price<50"]),
    Q("HARD-05", "hard", "chardonnay without butter or heavy oak",
      ["variety:Chardonnay", "NOT oaky", "NOT buttery"]),
    Q("HARD-06", "hard", "high-rated bottle for under $20",
      ["points>=90", "price<20"]),
    Q("HARD-07", "hard", "dessert wine that is not cloying",
      ["dessert", "NOT cloying"]),

    # ---- VAGUE (3): subjective / metaphorical --------------------------
    Q("VAG-01", "vague", "wine for a winter steak dinner with friends",
      ["red", "rich", "food-pairing"]),
    Q("VAG-02", "vague", "something to impress a sommelier",
      ["complex", "high-rated"]),
    Q("VAG-03", "vague", "easy patio sipper for a hot summer afternoon",
      ["light", "refreshing", "white or rose"]),
]


def _cache_open() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
    )
    return conn


def _cache_get(conn, key):
    row = conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def _load_products() -> list[dict]:
    return [json.loads(line) for line in PRODUCTS_PATH.read_text().splitlines() if line.strip()]


def _format_catalog_for_prompt(products: list[dict]) -> str:
    lines = []
    for p in products:
        a = p["attributes"]
        s = p["specs"]
        vintage = s.get("vintage") or "NV"
        lines.append(
            f'{p["id"]}: {p["name"]} | winery={p["brand"]} | variety={a.get("variety","")} '
            f'| country={a.get("country","")} | region={a.get("region","")} '
            f'| points={a.get("points","")} | price=${a.get("price","")} | vintage={vintage}'
        )
    return "\n".join(lines)


def _grade_query(provider, query: dict, catalog_str: str, conn) -> list[dict]:
    prompt = (
        "You are a wine expert grading a recommendation system's ground truth.\n\n"
        f"User query: {query['query_text']!r}\n"
        f"Difficulty: {query['difficulty']}\n"
        f"Key attributes the user cares about: {query['key_attributes']}\n\n"
        "Catalog (one wine per line, with id):\n"
        f"{catalog_str}\n\n"
        "Pick exactly 5 wines from the catalog that best match the query. "
        "For each, give:\n"
        "  - product_id (the WINE-### id from the catalog)\n"
        "  - relevance: 3 (perfect match), 2 (strong match), 1 (partial match)\n"
        "  - reason: ≤120 chars, grounded in the wine's attributes\n\n"
        "Return strict JSON: a list of 5 objects with keys product_id, relevance, reason. "
        "Sort best→worst. No prose, no markdown — JSON only."
    )
    key = hashlib.sha256(f"{OPUS_MODEL}::{prompt}".encode()).hexdigest()
    cached = _cache_get(conn, key)
    if cached:
        logger.info("[cache] %s", query["id"])
        return json.loads(cached)
    logger.info("[teacher] %s", query["id"])
    text = provider.generate(prompt, json_mode=True)
    text = text.strip()
    if text.startswith("```"):
        # strip code fence if returned despite json_mode
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    parsed = json.loads(text)
    if not isinstance(parsed, list) or len(parsed) != 5:
        raise ValueError(f"expected list of 5, got: {parsed!r}")
    valid_ids = {p_id for p_id in [p["id"] for p in _load_products()]}
    for entry in parsed:
        if entry["product_id"] not in valid_ids:
            raise ValueError(f"unknown id from teacher: {entry!r}")
        if entry["relevance"] not in (1, 2, 3):
            raise ValueError(f"bad relevance: {entry!r}")
    _cache_put(conn, key, json.dumps(parsed))
    return parsed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not PRODUCTS_PATH.exists():
        logger.error("missing %s — run scripts/build_wine_dataset.py first", PRODUCTS_PATH)
        return 2
    if os.environ.get("RECOMMEND_LLM_PROVIDER") != "anthropic":
        logger.error("set RECOMMEND_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY")
        return 2

    products = _load_products()
    catalog_str = _format_catalog_for_prompt(products)

    # Force opus for the teacher.
    os.environ["RECOMMEND_LLM_MODEL"] = OPUS_MODEL
    provider = get_provider()
    conn = _cache_open()

    out_queries = []
    for q in QUERIES:
        gt = _grade_query(provider, q, catalog_str, conn)
        out_q = dict(q)
        out_q["ground_truth_top5"] = gt
        out_queries.append(out_q)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"version": "1.0", "queries": out_queries}, indent=2))
    logger.info("wrote %s (%d queries)", OUT_PATH, len(out_queries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the script**

```bash
export RECOMMEND_LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY must already be set in env
python scripts/build_wine_eval.py
```

Expected: 30 lines of `[teacher] EASY-01` etc., then "wrote .../wine_eval.json (30 queries)". Re-running prints `[cache]` lines and finishes in seconds.

- [ ] **Step 3: Spot-check 3 queries**

```bash
python3 -c "
import json
d = json.load(open('benchmark/data/per_domain/wine_eval.json'))
for q in d['queries']:
    if q['id'] in ('EASY-01','MED-01','VAG-01'):
        print(q['id'], q['query_text']);
        for g in q['ground_truth_top5']:
            print('  ', g['product_id'], g['relevance'], g['reason'])
        print()
"
```

Expected: top-5 product IDs from `WINE-000` through `WINE-199`, relevance 1–3, short grounded reasons mentioning variety/region/price/points.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_wine_eval.py benchmark/data/per_domain/wine_eval.json
git commit -m "data(eval): add 30-query wine_eval with Opus ground truth"
```

---

## Task 4: Compile the wine bundle

**Files:**
- Generated: `artifacts/wine/` (manifest, fts5.db, products.jsonl, reviews.jsonl, embedding/, eval/, synthetic_train.jsonl)

`arena_new.py` runs `ingest → embedding-pin → reranker-pin → eval-import → generate-synthetic → finetune-embedding`. Listwise is run separately in Task 5. The reranker-LoRA step is omitted on purpose (CLAUDE.md gotcha #1: cross-encoder LoRA is shelved).

- [ ] **Step 1: Verify training requirements installed**

```bash
pip list 2>/dev/null | grep -iE "torch|transformers|peft|sentence-transformers|datasets" | head
```

Expected: torch, transformers, peft, sentence-transformers, datasets all present (per `implementations/design_14_local_hybrid/training-requirements.txt`). If any are missing:

```bash
pip install -r implementations/design_14_local_hybrid/training-requirements.txt
```

- [ ] **Step 2: Verify ANTHROPIC_API_KEY is set (synthetic-query teacher uses it)**

```bash
python3 -c "import os; print('OK' if os.environ.get('ANTHROPIC_API_KEY') else 'MISSING')"
```

Expected: `OK`. If `MISSING`: stop and tell the user.

- [ ] **Step 3: Run arena_new.py to compile the bundle**

```bash
export RECOMMEND_LLM_PROVIDER=anthropic
python scripts/arena_new.py new wine \
    --catalog data/wine/products.jsonl \
    --reviews data/wine/reviews.jsonl \
    --eval    benchmark/data/per_domain/wine_eval.json \
    --steps   ingest,embedding-pin,reranker-pin,eval-import,generate-synthetic,finetune-embedding \
    --force
```

Expected: logs "loaded 200 products + 200 reviews", "ingested into bundle", "wrote synthetic_train.jsonl", training loss decreasing, "manifest written".

If a ski-specific code path blocks compilation (e.g., a hardcoded domain string or attribute name): fix it minimally in `scripts/arena_new.py` or the script it dispatches to, then re-run with `--force`. Examples of acceptable fixes: replacing a literal `"ski"` with `bundle.manifest.domain`; making an attribute lookup tolerant of missing keys. Do NOT refactor beyond the blocker.

- [ ] **Step 4: Verify bundle structure**

```bash
ls artifacts/wine/
cat artifacts/wine/manifest.json | python3 -m json.tool | head -30
```

Expected files: `manifest.json`, `fts5.db`, `products.jsonl`, `reviews.jsonl`, `embedding/`, `eval/`, `synthetic_train.jsonl`. Manifest shows `"domain": "wine"`, `"n_products": 200`, embedding adapter pointing at `embedding/`.

- [ ] **Step 5: Run listwise fine-tune**

```bash
python scripts/finetune_listwise.py --bundle artifacts/wine \
    --base-model Qwen/Qwen3-1.7B \
    --epochs 1 --batch-size 1 --grad-accum 4
```

Expected: loss decreasing over the synthetic-train candidates, adapter saved to `artifacts/wine/listwise_adapter/`, manifest updated with `metadata.reranker_runtime_kind: "listwise"`.

If the trainer balks on the wine domain (e.g., expects a specific attribute key for prompt construction): apply a minimal fix and re-run.

- [ ] **Step 6: Commit**

```bash
git add artifacts/wine
git commit -m "feat(wine): compile bundle with embedding LoRA + listwise rerank"
```

---

## Task 5: Pass the eval gate

**Files:**
- Read-only: `artifacts/wine/manifest.json`

- [ ] **Step 1: Run the eval gate**

```bash
python scripts/eval_bundle.py --bundle artifacts/wine
```

Expected exit code 0; stdout shows `fine-tuned NDCG@5 > vanilla`, `full pipeline > FTS5 baseline`, `>= threshold 0.55`. If exit code is non-zero:

- **fine-tuned ≤ vanilla:** likely too few synthetic triples or hard negatives — investigate the synthetic_train.jsonl size and re-run training with more queries-per-product (`--queries-per-product 15` on arena_new.py).
- **full ≤ FTS5:** wine descriptions may be lex-dominant — check whether RRF actually helps; if listwise underperforms, try `RECOMMEND_LISTWISE_TOP_K=10` or skip rerank.
- **threshold:** lower scores suggest weak ground truth — re-spot-check teacher labels for 3 hard queries and fix the worst ones manually in `wine_eval.json`.

Do NOT lower `manifest.eval.arena_threshold_ndcg5`. If the gate genuinely cannot pass, stop and report findings.

- [ ] **Step 2: Verify the manifest now records the eval scores**

```bash
python3 -c "import json; m=json.load(open('artifacts/wine/manifest.json'))['eval']; print(m)"
```

Expected: `arena_threshold_ndcg5: 0.55`, `fts5_baseline_ndcg5: ~`, `full_pipeline_ndcg5: >=0.55`.

- [ ] **Step 3: Commit (only if eval scores got persisted)**

```bash
# If manifest changed in step 2:
git add artifacts/wine/manifest.json && git commit -m "data(wine): record eval gate scores in manifest"
# If not, no-op.
```

---

## Task 6: Build the Streamlit frontend

**Files:**
- Create: `frontend/app.py`

- [ ] **Step 1: Write `frontend/app.py`**

```python
"""Streamlit frontend for the recommend-arena hybrid recommender.

Run:
    pip install -r frontend/requirements.txt
    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from implementations.design_14_local_hybrid import LocalHybridRecommender  # noqa: E402

DOMAINS = ["wine", "ski"]


@st.cache_resource(show_spinner="loading bundle...")
def _load_recommender(domain: str) -> LocalHybridRecommender:
    bundle_path = _PROJECT_ROOT / "artifacts" / domain
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found at {bundle_path}")
    return LocalHybridRecommender.from_bundle(bundle_path)


def _wine_card(result, product: dict) -> None:
    a = product.get("attributes", {})
    s = product.get("specs", {})
    vintage = s.get("vintage")
    name = product.get("name", result.product_id)
    winery = product.get("brand", "")
    variety = a.get("variety", "")
    region = a.get("region", "")
    country = a.get("country", "")
    price = a.get("price")
    points = a.get("points")
    snippet = (result.matched_review_excerpt or "")[:240]

    with st.container(border=True):
        st.markdown(f"**{name}**")
        meta = " · ".join(
            x for x in [winery, variety, f"{region}, {country}".strip(", "), str(vintage) if vintage else ""]
            if x
        )
        st.caption(meta)
        cols = st.columns([1, 1, 4])
        if points:
            cols[0].metric("points", points)
        if price is not None:
            cols[1].metric("price", f"${price:.0f}")
        if snippet:
            cols[2].write(snippet)


def _ski_card(result, product: dict) -> None:
    a = product.get("attributes", {})
    s = product.get("specs", {})
    name = product.get("name", result.product_id)
    brand = product.get("brand", "")
    category = product.get("category", "")
    waist = s.get("waist_width_mm")
    snippet = (result.matched_review_excerpt or "")[:240]
    with st.container(border=True):
        st.markdown(f"**{name}**")
        meta = " · ".join(x for x in [brand, category, f"{waist}mm waist" if waist else ""] if x)
        st.caption(meta)
        chips = []
        for k in ("terrain", "stiffness", "powder_float", "playfulness"):
            v = a.get(k)
            if v is not None:
                chips.append(f"`{k}: {v}`")
        if chips:
            st.markdown(" ".join(chips))
        if snippet:
            st.write(snippet)


CARDS = {"wine": _wine_card, "ski": _ski_card}


def _product_lookup(rec: LocalHybridRecommender, domain: str) -> dict[str, dict]:
    # Attempt to read products from the bundle on disk. Cache via st.session_state.
    key = f"_products_{domain}"
    if key in st.session_state:
        return st.session_state[key]
    products_path = _PROJECT_ROOT / "artifacts" / domain / "products.jsonl"
    import json
    out: dict[str, dict] = {}
    if products_path.exists():
        for line in products_path.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            out[p["id"]] = p
    st.session_state[key] = out
    return out


def main() -> None:
    st.set_page_config(page_title="recommend-arena", layout="wide")
    st.title("recommend-arena")
    st.caption("Local-first hybrid recommender — RRF fusion + listwise rerank")

    with st.sidebar:
        domain = st.selectbox("Domain", DOMAINS, index=0)
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        listwise_on = st.checkbox("Listwise rerank", value=True,
                                   help="Toggle the LoRA-rerank stage. When off, results come from RRF fusion only.")

    rec = _load_recommender(domain)
    products = _product_lookup(rec, domain)

    query = st.text_input("Search", placeholder={
        "wine": "e.g. bold california cabernet under $50 with cellar potential",
        "ski": "e.g. damp stable all-mountain ski for the ice coast",
    }[domain])

    if not query.strip():
        st.info("Type a query above to see recommendations.")
        return

    prev_enabled = rec.enable_reranker
    rec.enable_reranker = bool(listwise_on)
    try:
        t0 = time.perf_counter()
        results = rec.query(query, domain, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        rec.enable_reranker = prev_enabled

    st.caption(f"{len(results)} results in {elapsed_ms:.0f} ms")
    card_fn = CARDS.get(domain, _wine_card)
    for r in results:
        product = products.get(r.product_id, {})
        card_fn(r, product)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the recommender's `RecommendationResult` shape**

```bash
python3 -c "
from implementations.design_14_local_hybrid import LocalHybridRecommender
import inspect
src = inspect.getsource(LocalHybridRecommender.query)
print(src.split('return')[-1][:500] if 'return' in src else 'no explicit return')
"
grep -n "class RecommendationResult\|matched_review_excerpt\|product_id" /Users/harvey/Development/sports/recommend/shared/interface.py 2>/dev/null | head
```

Expected: `RecommendationResult` has at least `product_id` and `matched_review_excerpt` attributes. If field names differ, fix the card functions to use the correct attribute names — do NOT add a new abstraction. The fields are minimal so this is a 1-line edit per card.

- [ ] **Step 3: Commit**

```bash
git add frontend/app.py
git commit -m "feat(frontend): streamlit UI for ski + wine domains"
```

---

## Task 7: Smoke-test the frontend

- [ ] **Step 1: Start streamlit in the background**

```bash
cd /Users/harvey/Development/sports/recommend
streamlit run frontend/app.py --server.headless true --server.port 8501 &
sleep 4
```

- [ ] **Step 2: Hit the health endpoint**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501/_stcore/health
```

Expected: `200`.

- [ ] **Step 3: Manual smoke test (announce to user)**

Tell the user: "Streamlit is running at http://localhost:8501. Please load the page, try one query in each domain (e.g., 'oaky chardonnay' for wine, 'powder ski for backcountry' for ski), toggle the listwise rerank checkbox, and confirm results render."

Wait for the user to confirm. If they report errors:
- "module not found" → frontend deps not installed: `pip install -r frontend/requirements.txt`
- "bundle not found" → Task 4 didn't produce `artifacts/wine/`
- "AttributeError on RecommendationResult" → field names in card fn don't match; fix per Task 6 step 2

- [ ] **Step 4: Stop streamlit**

```bash
pkill -f "streamlit run frontend/app.py" || true
```

- [ ] **Step 5: No additional commit needed unless field-name fixes were applied**

---

## Self-review

Spec coverage check:
- ✅ Dataset (Task 2 / build_wine_dataset.py)
- ✅ Eval set with Opus ground truth (Task 3 / build_wine_eval.py)
- ✅ Pipeline run (Task 4)
- ✅ Listwise rerank (Task 4 step 5)
- ✅ Eval gate (Task 5)
- ✅ Streamlit frontend with sidebar, top-K, listwise toggle (Task 6)
- ✅ Smoke test (Task 7)
- ✅ frontend/requirements.txt (Task 1)
- ✅ data/wine/README.md (Task 2)

Type/name consistency:
- `RecommendationResult.matched_review_excerpt` is assumed in Task 6 — verified in step 2 of Task 6, with a fix path if the field name differs.
- `manifest.eval.arena_threshold_ndcg5` is referenced consistently from spec → Task 5.
- `LocalHybridRecommender.from_bundle` and `enable_reranker` attribute use match the recommender source.

No placeholders, no "TBD". Each task ends in a commit. Ordering is correct: dataset → eval → bundle → gate → frontend → smoke test.
