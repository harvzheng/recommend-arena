"""Design #13 — Frontier Listwise (Opus 4.7).

Strategy: format the entire per-domain catalog (products + their reviews) into
one text block, send it as a cached system prompt, ask Opus 4.7 to return the
top-k product IDs as a JSON array, parse, and return.

This is the arena's upper-bound reference. It exists to anchor the gap
between "frontier model with full context" and "what a local pipeline can
do". It is not a design we'd ship to production.

Cost (ski domain, ~28k input tokens):
- First query (cache miss):  ~$0.43
- Subsequent (cache hit):    ~$0.05
- Full 20-query benchmark:   ~$1.40

Caveats handled:
- The model occasionally emits IDs outside the catalog. We filter to known
  IDs and pad with a deterministic fallback so the runner always gets top_k.
- Per-query results are cached locally (SQLite) so re-running the benchmark
  is free. Disable with env ARENA_NO_CACHE=1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.interface import RecommendationResult  # noqa: E402

logger = logging.getLogger(__name__)

# Pinned per the claude-api skill model table. Do not append a date suffix.
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 1024

# Local response cache so repeated benchmark runs are free and deterministic.
# The benchmark runner re-issues each query num_runs times; without this we'd
# pay 3x cost per run for the same answer.
_DEFAULT_CACHE_PATH = (
    _PROJECT_ROOT / "benchmark" / "results" / "design_13" / "cache.sqlite"
)


class OpusListwiseRecommender:
    """Recommender that delegates ranking to Opus 4.7 over the full catalog."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_path: str | Path | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # domain -> formatted catalog string (sent as a cached system block)
        self._catalogs: dict[str, str] = {}
        # domain -> list of known product IDs in deterministic order
        self._known_ids: dict[str, list[str]] = {}
        # domain -> {product_id: product_name}
        self._product_names: dict[str, dict[str, str]] = {}

        self._client = None  # lazy — only build when a query needs the API

        # Local cache
        if os.environ.get("ARENA_NO_CACHE") == "1":
            self._cache = None
        else:
            path = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE_PATH
            self._cache = _ResultCache(path)

    # ------------------------------------------------------------------
    # Recommender protocol
    # ------------------------------------------------------------------
    def ingest(
        self, products: list[dict], reviews: list[dict], domain: str
    ) -> None:
        """Format the catalog into a single cacheable text block.

        No API calls happen here — the formatted string is sent at query time
        inside a `cache_control: ephemeral` block so the second and subsequent
        queries against the same domain hit the prompt cache.
        """
        reviews_by_product: dict[str, list[dict]] = {}
        for r in reviews:
            pid = r.get("product_id") or r.get("id") or ""
            if pid:
                reviews_by_product.setdefault(pid, []).append(r)

        # Sort products by ID for deterministic prompt bytes — required for the
        # prompt cache to actually hit on subsequent queries.
        sorted_products = sorted(
            products,
            key=lambda p: p.get("id") or p.get("product_id") or p.get("name", ""),
        )

        names: dict[str, str] = {}
        ids: list[str] = []
        for p in sorted_products:
            pid = p.get("id") or p.get("product_id") or ""
            if pid:
                ids.append(pid)
                names[pid] = p.get("name") or p.get("product_name") or pid

        self._catalogs[domain] = self._format_catalog(
            sorted_products, reviews_by_product
        )
        self._known_ids[domain] = ids
        self._product_names[domain] = names

        logger.info(
            "design_13: ingested %d products + %d reviews for domain %r (%d chars)",
            len(products), len(reviews), domain, len(self._catalogs[domain]),
        )

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        if domain not in self._catalogs:
            logger.warning("design_13: domain %r not ingested", domain)
            return []

        catalog = self._catalogs[domain]
        known_ids = self._known_ids[domain]
        names = self._product_names[domain]

        cache_key = _hash_key(self.model, domain, query_text, top_k, catalog)
        cached_ids: list[str] | None = None
        if self._cache is not None:
            cached_ids = self._cache.get(cache_key)

        if cached_ids is not None:
            ids = cached_ids
        else:
            ids = self._call_api(query_text, domain, top_k)
            if self._cache is not None:
                self._cache.put(cache_key, ids)

        # Filter hallucinated IDs and pad to top_k from a deterministic fallback.
        valid_set = set(known_ids)
        seen: set[str] = set()
        ranked: list[str] = []
        for pid in ids:
            if pid in valid_set and pid not in seen:
                ranked.append(pid)
                seen.add(pid)
        if len(ranked) < top_k:
            for pid in known_ids:
                if pid not in seen:
                    ranked.append(pid)
                    seen.add(pid)
                    if len(ranked) >= top_k:
                        break

        results: list[RecommendationResult] = []
        for i, pid in enumerate(ranked[:top_k]):
            score = max(0.05, 1.0 - i * 0.05)
            results.append(RecommendationResult(
                product_id=pid,
                product_name=names.get(pid, pid),
                score=round(score, 4),
                explanation=(
                    f"Ranked #{i + 1} by Opus 4.7 listwise over the full "
                    f"{domain} catalog."
                ),
                matched_attributes={},
            ))
        return results

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------
    def _call_api(self, query_text: str, domain: str, top_k: int) -> list[str]:
        """Send the cached catalog + query and parse the returned ID list."""
        client = self._get_client()
        catalog = self._catalogs[domain]

        system_blocks = [{
            "type": "text",
            "text": catalog,
            "cache_control": {"type": "ephemeral"},
        }]
        user_prompt = (
            f"Query: {query_text}\n\n"
            f"Return the top-{top_k} most relevant product IDs as a JSON "
            f"array, ordered best-first. Use ONLY IDs that appear in the "
            f"catalog above. Output the JSON array only, no prose."
        )

        try:
            # Opus 4.7: omit `thinking`, `temperature`, `top_p`, `top_k`.
            # Listwise ranking is deterministic enough; we don't need thinking.
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            logger.error("design_13: API call failed: %s", e)
            return []

        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "design_13: usage in=%s cache_w=%s cache_r=%s out=%s",
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "cache_creation_input_tokens", "?"),
                getattr(usage, "cache_read_input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
            )

        text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break

        return _parse_id_array(text)

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "design_13 requires the `anthropic` Python package. "
                    "Install with: pip install anthropic"
                ) from e
            if not self._api_key:
                raise RuntimeError(
                    "design_13 requires ANTHROPIC_API_KEY in the environment."
                )
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Catalog formatting
    # ------------------------------------------------------------------
    def _format_catalog(
        self,
        products: list[dict],
        reviews_by_product: dict[str, list[dict]],
    ) -> str:
        """Render the catalog as one stable text block.

        Determinism matters here — the prompt cache is a byte-prefix match.
        Any nondeterminism in this function (set iteration order, dict
        re-ordering) silently kills cache hits.
        """
        lines: list[str] = [
            "You are ranking products in a single-domain catalog. Below is "
            "the complete catalog: every product and every review. When the "
            "user gives you a query, return the most relevant product IDs.",
            "",
            "=== CATALOG ===",
        ]
        for p in products:
            pid = p.get("id") or p.get("product_id") or ""
            name = p.get("name") or p.get("product_name") or pid
            brand = p.get("brand", "")
            category = p.get("category", "")

            lines.append("")
            lines.append(f"## {pid} — {name}")
            if brand or category:
                lines.append(f"   brand: {brand}    category: {category}")

            specs = p.get("specs") or {}
            if specs:
                lines.append("   specs:")
                for k in sorted(specs.keys()):
                    lines.append(f"     - {k}: {_compact(specs[k])}")

            attrs = p.get("attributes") or {}
            if attrs:
                lines.append("   attributes:")
                for k in sorted(attrs.keys()):
                    lines.append(f"     - {k}: {_compact(attrs[k])}")

            reviews = reviews_by_product.get(pid, [])
            if reviews:
                lines.append("   reviews:")
                # Sort reviews deterministically too
                for r in sorted(
                    reviews,
                    key=lambda x: (
                        x.get("reviewer", ""),
                        (x.get("text") or x.get("review_text") or "")[:64],
                    ),
                ):
                    rev_text = (r.get("text") or r.get("review_text") or "").strip()
                    rating = r.get("rating", "")
                    rating_str = f" [{rating}/5]" if rating != "" else ""
                    lines.append(f"     - ({r.get('reviewer', 'anon')}{rating_str}) {rev_text}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def _parse_id_array(text: str) -> list[str]:
    """Extract a JSON array of strings from arbitrary model output."""
    if not text:
        return []
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass
    # Fall back to first array-shaped substring
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            return []
    return []


def _compact(value) -> str:
    """Compact one attribute value for prompt printing."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _hash_key(model: str, domain: str, query: str, top_k: int, catalog: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(domain.encode())
    h.update(b"\0")
    h.update(query.encode())
    h.update(b"\0")
    h.update(str(top_k).encode())
    h.update(b"\0")
    h.update(hashlib.sha256(catalog.encode()).digest())
    return h.hexdigest()


class _ResultCache:
    """Thread-safe SQLite-backed cache mapping query hash → ranked ID list."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._local = threading.local()
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS query_cache ("
            "  key TEXT PRIMARY KEY,"
            "  ids_json TEXT NOT NULL,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path)
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return c

    def get(self, key: str) -> list[str] | None:
        row = self._conn().execute(
            "SELECT ids_json FROM query_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            return None
        return None

    def put(self, key: str, ids: list[str]) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO query_cache (key, ids_json) VALUES (?, ?)",
            (key, json.dumps(ids)),
        )
        c.commit()
