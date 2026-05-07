"""Streamlit frontend for the recommend-arena hybrid recommender.

Run:
    pip install -r frontend/requirements.txt
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from implementations.design_14_local_hybrid import LocalHybridRecommender  # noqa: E402

DOMAINS = ["wine", "ski"]
PLACEHOLDERS = {
    "wine": "e.g. bold california cabernet under $50 with cellar potential",
    "ski": "e.g. damp stable all-mountain ski for the ice coast",
}


@st.cache_resource(show_spinner="loading bundle...")
def _load_recommender(domain: str) -> LocalHybridRecommender:
    bundle_path = _PROJECT_ROOT / "artifacts" / domain
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found at {bundle_path}")
    return LocalHybridRecommender.from_bundle(bundle_path)


@st.cache_data(show_spinner=False)
def _load_products(domain: str) -> dict[str, dict]:
    path = _PROJECT_ROOT / "artifacts" / domain / "products.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                out[p["id"]] = p
    return out


@st.cache_data(show_spinner=False)
def _load_reviews(domain: str) -> dict[str, list[str]]:
    path = _PROJECT_ROOT / "artifacts" / domain / "reviews.jsonl"
    out: dict[str, list[str]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["product_id"], []).append(r.get("text", ""))
    return out


def _wine_card(result, product: dict, review_text: str) -> None:
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
    snippet = (review_text or "")[:280]

    with st.container(border=True):
        st.markdown(f"**{name}**")
        meta_parts = []
        if winery:
            meta_parts.append(winery)
        if variety:
            meta_parts.append(variety)
        loc = ", ".join(x for x in [region, country] if x)
        if loc:
            meta_parts.append(loc)
        if vintage:
            meta_parts.append(str(vintage))
        st.caption(" · ".join(meta_parts))
        cols = st.columns([1, 1, 6])
        if points:
            cols[0].metric("points", points)
        if price is not None:
            cols[1].metric("price", f"${price:.0f}")
        if snippet:
            cols[2].write(snippet)
        if result.explanation:
            st.caption(f"why: {result.explanation}")


def _ski_card(result, product: dict, review_text: str) -> None:
    a = product.get("attributes", {})
    s = product.get("specs", {})
    name = product.get("name", result.product_id)
    brand = product.get("brand", "")
    category = product.get("category", "")
    waist = s.get("waist_width_mm")
    snippet = (review_text or "")[:280]
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
        if result.explanation:
            st.caption(f"why: {result.explanation}")


CARDS = {"wine": _wine_card, "ski": _ski_card}


def main() -> None:
    st.set_page_config(page_title="recommend-arena", layout="wide")
    st.title("recommend-arena")
    st.caption("Local-first hybrid recommender — RRF fusion + listwise rerank")

    available_domains = [d for d in DOMAINS if (_PROJECT_ROOT / "artifacts" / d).exists()]
    if not available_domains:
        st.error(
            f"No bundles found in {_PROJECT_ROOT}/artifacts/. "
            "Run `python scripts/arena_new.py new <domain> ...` first."
        )
        return

    with st.sidebar:
        domain = st.selectbox("Domain", available_domains, index=0)
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        listwise_on = st.checkbox(
            "Listwise rerank", value=True,
            help="Toggle the LoRA-rerank stage. When off, results come from RRF fusion only.",
        )

    rec = _load_recommender(domain)
    products = _load_products(domain)
    reviews = _load_reviews(domain)

    query = st.text_input("Search", placeholder=PLACEHOLDERS[domain])

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
        review_text = ""
        for t in reviews.get(r.product_id, []):
            if t:
                review_text = t
                break
        card_fn(r, product, review_text)


if __name__ == "__main__":
    main()
