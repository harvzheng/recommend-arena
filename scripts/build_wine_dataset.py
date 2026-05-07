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

HF_DATASET = "spawn99/wine-reviews"

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
    rng = np.random.default_rng(seed)
    by_variety = defaultdict(list)
    for r in rows:
        by_variety[r["variety"]].append(r)
    by_country = defaultdict(list)
    for r in rows:
        by_country[r["country"]].append(r)

    chosen: list[dict] = []
    chosen_ids: set[int] = set()

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

    high_pts = [r for r in rows if r["points"] >= 88]
    by_tier = defaultdict(list)
    for r in high_pts:
        by_tier[_price_tier(r["price"])].append(r)
    target_per_tier = n // 3
    for tier in ("low", "mid", "high"):
        pool = list(by_tier[tier])
        rng.shuffle(pool)
        for r in pool:
            if len(chosen) >= n:
                break
            tier_count = sum(1 for c in chosen if _price_tier(c["price"]) == tier)
            if tier_count >= target_per_tier and len(chosen) < n:
                continue
            if id(r) not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(id(r))

    if len(chosen) < n:
        pool = list(high_pts)
        rng.shuffle(pool)
        for r in pool:
            if len(chosen) >= n:
                break
            if id(r) not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(id(r))

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
        "rating": max(1, min(5, round((r["points"] - 80) / 4))),
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
