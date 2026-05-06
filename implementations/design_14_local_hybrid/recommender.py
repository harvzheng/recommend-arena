"""Design #14 — Local-First Hybrid (the runtime).

Pipeline:
    1. filter parser (deterministic regex)
    2. hard prefilter (Rust: SQL WHERE assembly)
    3. parallel retrieval (Rust FTS5 + Python vector encoder)
    4. RRF fusion (Rust)
    5. cross-encoder rerank (Python; optional)
    6. explanation (deterministic)

Hot path (steps 2, 3-FTS5, 4) lives in `arena_core` (Rust + PyO3).
Steps 3-vector and 5 stay in Python because they invoke ML models.

Inference tier 1 only in this slice (14.0). Tiers 2 and 3 (LLM filter
parser, frontier escalation) are wired in a follow-up.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.interface import RecommendationResult  # noqa: E402

from . import retrieval  # noqa: E402
from .filter_parser import parse_query, tokenize_for_fts5  # noqa: E402
from .rerank import rerank  # noqa: E402
from . import listwise_rerank as _listwise  # noqa: E402
from .rrf import backend_name, rrf_fuse  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables — kept as module-level constants so the eval gate can sweep them
# without touching the orchestrator.
# ---------------------------------------------------------------------------
TOP_K_LEXICAL = 100
TOP_K_VECTOR = 100
TOP_K_FUSED = 50
TOP_K_RERANKED = 10
RRF_K = 60

# Per-query routing: skip the vector+rerank stage when lex top-1 has a
# meaningful absolute BM25 score AND also appears in the vector retrieval
# top-K. Two retrievers agreeing on the same top is a strong signal that
# the answer is "lexically obvious"; downstream rerank tends to *hurt*
# such queries by mixing in semantically-similar wrong candidates.
LEX_DOMINANCE_MIN_SCORE = float(
    os.environ.get("RECOMMEND_LEX_DOMINANCE_MIN_SCORE", "1.0")
)
LEX_DOMINANCE_VEC_K = int(
    os.environ.get("RECOMMEND_LEX_DOMINANCE_VEC_K", "10")
)
# Lex top-1 pinning: when FTS5's top-1 has a strong absolute score we
# promote it to position 1 of the final ranked output regardless of
# what rerank decided. Targets the VAGUE-02 case (lex top-1 = gold
# rel-3, vector retrieval misses it entirely, rerank demotes it).
LEX_PIN_MIN_SCORE = float(
    os.environ.get("RECOMMEND_LEX_PIN_MIN_SCORE", "2.0")
)


class LocalHybridRecommender:
    """Design 14 runtime."""

    def __init__(
        self,
        db_dir: str | None = None,
        embedding_model: str | None = None,
        embedding_adapter_path: str | None = None,
        reranker_model: str | None = None,
        reranker_adapter_path: str | None = None,
        reranker_kind: str = "cross_encoder",  # "cross_encoder" | "listwise"
        enable_reranker: bool = True,
        enable_vector: bool = True,
    ):
        if db_dir is None:
            # Per-instance temp dir keeps test runs isolated. Production use
            # should pass a path that survives across runs so the FTS5 index
            # doesn't have to be rebuilt every time.
            self._tmpdir = tempfile.TemporaryDirectory(prefix="design14_")
            db_dir = self._tmpdir.name
        else:
            self._tmpdir = None
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model or retrieval.DEFAULT_EMBED_MODEL
        self.embedding_adapter_path = embedding_adapter_path
        self.reranker_model = reranker_model
        self.reranker_adapter_path = reranker_adapter_path
        self.reranker_kind = reranker_kind
        self.enable_reranker = enable_reranker
        self.enable_vector = enable_vector

        # Per-domain state
        self._domains: dict[str, _DomainState] = {}
        self._lock = threading.Lock()

        logger.info("design_14: backend=%s db_dir=%s", backend_name(), self.db_dir)

    # ------------------------------------------------------------------
    # Bundle integration
    # ------------------------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle_path: str | Path) -> "LocalHybridRecommender":
        """Instantiate the recommender from an `artifacts/<domain>/` bundle.

        The bundle's manifest pins the embedding and reranker model IDs;
        the FTS5 index is loaded directly from the bundle's `fts5.db`.
        Products and reviews are re-read from the bundle so the in-memory
        `_DomainState` stays consistent with what the eval gate sees.
        """
        # Imported here to avoid circular imports at module-load time.
        from shared.domain_bundle import Bundle  # noqa: WPS433

        bundle = Bundle.load(bundle_path)
        domain = bundle.manifest.domain

        emb = bundle.manifest.embedding
        rer = bundle.manifest.reranker

        embedding_model = emb.base_model if emb else retrieval.DEFAULT_EMBED_MODEL
        embedding_adapter_path = (
            str(bundle.paths.root / emb.adapter_path)
            if emb and emb.adapter_path and emb.kind != "off_the_shelf"
            else None
        )
        reranker_model = rer.base_model if rer else None
        reranker_adapter_path = (
            str(bundle.paths.root / rer.adapter_path)
            if rer and rer.adapter_path and rer.kind != "off_the_shelf"
            else None
        )
        # Manifest carries an explicit kind ("listwise" or "cross_encoder")
        # via metadata.reranker_runtime_kind. Default to cross_encoder when
        # missing (backwards compat with bundles trained before listwise).
        reranker_kind = (
            bundle.manifest.metadata.get("reranker_runtime_kind") or "cross_encoder"
        )

        # Honor an explicit disable from the manifest. We set this when the
        # eval matrix shows the rerank stage is net-negative on top of a
        # strong embedding (the canonical case: Qwen3-Embedding-0.6B + RRF
        # is the ski-bundle ceiling at 0.614; any rerank stage drops it).
        enable_reranker = (
            bool(bundle.manifest.metadata.get("enable_reranker", True))
        )

        # Use the bundle's directory as the db_dir so the runtime reuses
        # the FTS5 index that `arena new` already built. Do NOT clobber
        # it — copy the data files in via Bundle.read_*.
        rec = cls(
            db_dir=str(bundle.paths.root),
            embedding_model=embedding_model,
            embedding_adapter_path=embedding_adapter_path,
            reranker_model=reranker_model,
            reranker_adapter_path=reranker_adapter_path,
            reranker_kind=reranker_kind,
            enable_reranker=enable_reranker,
        )
        # Re-ingest into the bundle's directory. open_db is idempotent
        # and will reuse the FTS5 db that's already there.
        products = bundle.read_products()
        reviews = bundle.read_reviews()
        rec.ingest(products, reviews, domain)
        return rec

    # ------------------------------------------------------------------
    # Recommender protocol
    # ------------------------------------------------------------------
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Build the per-domain SQLite + FTS5 index and encode product vectors."""
        db_path = self.db_dir / f"{_safe_name(domain)}.sqlite"
        conn = retrieval.open_db(str(db_path))

        try:
            id_map = retrieval.ingest_products(conn, products, reviews)
        finally:
            conn.close()  # close so Rust can open it cleanly read-only

        product_names = {
            p.get("id") or p.get("product_id"): p.get("name") or p.get("product_name")
            for p in products
            if (p.get("id") or p.get("product_id"))
        }

        product_vecs: dict[str, list[float]] = {}
        if self.enable_vector:
            product_vecs = retrieval.encode_products(
                products, reviews,
                model_name=self.embedding_model,
                adapter_path=self.embedding_adapter_path,
            )

        # Prepare content snippets used by the cross-encoder rerank stage.
        # We don't want to build these on every query.
        rerank_docs = _build_rerank_docs(products, reviews)

        self._domains[domain] = _DomainState(
            db_path=str(db_path),
            id_map=id_map,
            product_names=product_names,
            product_vecs=product_vecs,
            rerank_docs=rerank_docs,
        )
        logger.info(
            "design_14: ingested domain=%s products=%d reviews=%d "
            "vector_track=%s",
            domain, len(products), len(reviews), bool(product_vecs),
        )

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        state = self._domains.get(domain)
        if state is None:
            logger.warning("design_14: domain %r not ingested", domain)
            return []

        # ----- Stage 1: filter parser -----
        all_filters = parse_query(query_text, domain)
        # Split out negated filters — the OR-based Rust prefilter can't
        # narrow on a NOT condition (its EXISTS-subqueries are joined
        # with OR), so negation runs as a post-retrieval AND-exclusion
        # against the retrieved candidate set.
        filters = [f for f in all_filters if not f.get("_negate")]
        negative_filters = [
            {k: v for k, v in f.items() if k != "_negate"}
            for f in all_filters if f.get("_negate")
        ]

        # ----- Stage 2: hard prefilter (RUST) -----
        candidate_ids = self._prefilter_candidates(state, filters)
        if negative_filters:
            excluded_ids = self._excluded_ids(state, negative_filters)
            if excluded_ids:
                candidate_ids = [c for c in candidate_ids if c not in excluded_ids]

        # ----- Stage 3: parallel retrieval -----
        lexical_ranked = self._fts5_track(state, query_text, candidate_ids)
        vector_ranked = self._vector_track(state, query_text, candidate_ids)

        # ----- Stage 3b: per-query routing -----
        # Skip the vector+rerank stages when lex top-1 has a strong
        # absolute score AND vector retrieval also surfaces it in its
        # top-K. Two retrievers agreeing is a high-precision signal that
        # the answer is "lexically obvious"; the cross-encoder rerank
        # stage destroys such queries by mixing in semantically-similar
        # wrong candidates (VAGUE-01, EASY-04, MED-05 on the skis bench).
        #
        # Listwise rerank is excluded — bypassing on a lex-dominance
        # gate trades a few wins (VAGUE-02 0.49→1.00) for bigger losses
        # on queries where the listwise reranker was actively helping
        # (VAGUE-13 0.63→0.31, VAGUE-19 0.83→0.18). Net delta on the
        # 100-query bench: -0.058 with the bypass enabled. The 4B model
        # is good enough to keep on the hot path.
        routing_eligible = (
            self.enable_reranker
            and self.reranker_kind != "listwise"
            and self._lex_dominates(lexical_ranked, vector_ranked)
        )
        if routing_eligible:
            top_pids = [pid for pid, _ in lexical_ranked[:max(top_k, TOP_K_RERANKED)]]
            return self._build_results(
                top_pids, query_text, filters, lexical_ranked, vector_ranked,
                state, top_k, lex_routed=True,
            )

        # ----- Stage 4: RRF fusion (RUST) -----
        ranked_lists = []
        if lexical_ranked:
            ranked_lists.append([pid for pid, _ in lexical_ranked])
        if vector_ranked:
            ranked_lists.append([pid for pid, _ in vector_ranked])

        if not ranked_lists:
            # Both retrievers returned nothing — fall back to the candidate
            # set in original catalog order so we still return SOMETHING.
            ranked_lists = [list(candidate_ids)] if candidate_ids else [list(state.product_names.keys())]

        fused = rrf_fuse(ranked_lists, k=RRF_K, top_k=TOP_K_FUSED)
        if not fused:
            return []

        # ----- Stage 5: rerank (Python; optional) -----
        if self.enable_reranker:
            rerank_input = [
                (pid, state.rerank_docs.get(pid, state.product_names.get(pid, pid)))
                for pid, _score in fused
            ]
            if self.reranker_kind == "listwise":
                # Recall analysis on the ski benchmark: recall@10 = 0.71 vs
                # recall@25 = 0.93. Feed 20 candidates so we don't bottleneck
                # on retrieval recall before rerank can do its job.
                listwise_input_k = int(os.environ.get("RECOMMEND_LISTWISE_TOP_K", "20"))
                reranked = _listwise.rerank(
                    query_text,
                    rerank_input[:listwise_input_k],
                    top_k=max(top_k, TOP_K_RERANKED),
                    model_name=self.reranker_model or _listwise.DEFAULT_LISTWISE_MODEL,
                    adapter_path=self.reranker_adapter_path,
                )
            else:
                reranked = rerank(
                    query_text,
                    rerank_input,
                    top_k=max(top_k, TOP_K_RERANKED),
                    model_name=self.reranker_model or "BAAI/bge-reranker-v2-m3",
                )
        else:
            reranked = [(pid, score) for pid, score in fused[:max(top_k, TOP_K_RERANKED)]]

        # ----- Stage 5b: lex top-1 pinning (cross-encoder only) -----
        # When the lexical retriever has a strong absolute score for its
        # top-1 hit AND the rerank pipeline kept that product in its
        # top-3 (rerank somewhat agrees but demoted it), promote lex
        # top-1 to position 1. This handles the VAGUE-02 case where
        # vector retrieval missed the lex-perfect answer entirely so
        # consensus routing can't fire. The "rerank top-3" guard avoids
        # promoting products the reranker actively rejected.
        #
        # Skipped for listwise: listwise saw the full top-20 (including
        # lex top-1) and made its decision; overriding here loses the
        # listwise reordering wins.
        pin_eligible = (
            self.reranker_kind != "listwise"
            and lexical_ranked
            and lexical_ranked[0][1] >= LEX_PIN_MIN_SCORE
        )
        if pin_eligible:
            pin_pid = lexical_ranked[0][0]
            rerank_top3 = {p for p, _ in reranked[:3]}
            if reranked and reranked[0][0] != pin_pid and pin_pid in rerank_top3:
                rest = [(p, s) for p, s in reranked if p != pin_pid]
                top_score = reranked[0][1]
                reranked = [(pin_pid, top_score + 1e-3), *rest]

        # ----- Stage 6: explanation + result construction -----
        results: list[RecommendationResult] = []
        # Score normalization: linear-shift the reranker output to [0.05, 1.0]
        # so the runner sees the standard score range.
        if reranked:
            scores = [s for _, s in reranked]
            s_min, s_max = min(scores), max(scores)
            spread = s_max - s_min if s_max > s_min else 1.0
            for i, (pid, raw_score) in enumerate(reranked[:top_k]):
                norm = (raw_score - s_min) / spread
                norm = max(0.05, min(1.0, 0.05 + 0.95 * norm))
                results.append(RecommendationResult(
                    product_id=pid,
                    product_name=state.product_names.get(pid, pid),
                    score=round(norm, 4),
                    explanation=_build_explanation(filters, lexical_ranked, vector_ranked, pid),
                    matched_attributes=_matched_attrs(filters),
                ))
        return results

    # ------------------------------------------------------------------
    # Per-query routing
    # ------------------------------------------------------------------
    @staticmethod
    def _lex_dominates(
        lexical_ranked: list[tuple[str, float]],
        vector_ranked: list[tuple[str, float]],
    ) -> bool:
        """Two-retriever consensus check.

        Returns True when (a) lex top-1 has a meaningful absolute BM25
        score and (b) vector retrieval also surfaces the same product
        in its top-K. Two-retriever agreement is a high-precision signal
        that lex's top-1 is the correct answer.

        When vector retrieval is disabled (e.g. the lexical-only baseline
        configuration), we return False here so the caller continues to
        produce results in the standard way (no double counting).
        """
        if not lexical_ranked:
            return False
        s1 = lexical_ranked[0][1]
        if s1 < LEX_DOMINANCE_MIN_SCORE:
            return False
        if not vector_ranked:
            return False
        lex_top1 = lexical_ranked[0][0]
        vec_top_k_ids = {pid for pid, _ in vector_ranked[:LEX_DOMINANCE_VEC_K]}
        return lex_top1 in vec_top_k_ids

    def _build_results(
        self,
        ordered_pids: list[str],
        query_text: str,
        filters: list[dict],
        lexical_ranked: list[tuple[str, float]],
        vector_ranked: list[tuple[str, float]],
        state: "_DomainState",
        top_k: int,
        lex_routed: bool = False,
    ) -> list[RecommendationResult]:
        """Build RecommendationResult list. Used by both the full pipeline
        and the lexical-routing short-circuit."""
        results: list[RecommendationResult] = []
        n = len(ordered_pids[:top_k])
        for i, pid in enumerate(ordered_pids[:top_k]):
            # Geometric falloff so position-1 isn't always exactly 1.0,
            # mirroring the rerank-path normalization.
            norm = max(0.05, 1.0 - (0.95 * i / max(1, n - 1)) if n > 1 else 1.0)
            explain = _build_explanation(filters, lexical_ranked, vector_ranked, pid)
            if lex_routed:
                explain = (explain or "") + " [lex-routed: top-1 dominance]"
            results.append(RecommendationResult(
                product_id=pid,
                product_name=state.product_names.get(pid, pid),
                score=round(norm, 4),
                explanation=explain,
                matched_attributes=_matched_attrs(filters),
            ))
        return results

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    def _excluded_ids(
        self, state: "_DomainState", negative_filters: list[dict]
    ) -> set[str]:
        """Resolve the set of product external_ids that match ANY of the
        given negative filters. Used to apply post-retrieval AND-exclusion
        for "NOT X" / "without Y" query phrases that the OR-based Rust
        prefilter can't narrow on.

        Returns an empty set on any error so the pipeline degrades to
        "no exclusion" instead of dropping the whole query.
        """
        if not negative_filters:
            return set()
        try:
            import arena_core  # type: ignore
            if not hasattr(arena_core, "build_prefilter_sql"):
                return set()
            where_fragment, params = arena_core.build_prefilter_sql(negative_filters)
        except (ImportError, ValueError) as e:
            logger.debug("design_14: negative-filter assembly failed: %s", e)
            return set()
        if not where_fragment:
            return set()
        sql = f"SELECT p.external_id FROM products p WHERE {where_fragment}"
        conn = sqlite3.connect(state.db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            logger.debug("design_14: negative-filter SQL failed: %s", e)
            return set()
        finally:
            conn.close()
        return {r[0] for r in rows}

    def _prefilter_candidates(
        self, state: "_DomainState", filters: list[dict]
    ) -> list[str]:
        """Run the Rust-built prefilter against the on-disk DB.

        Returns the list of candidate external_ids. When the prefilter
        is empty or matches nothing, we fall back to the full domain
        catalog — letting the retrieval + rerank stages do all the work.
        """
        # When there are no filters, there's no work for Rust to do.
        # Skip the SQL round-trip entirely.
        if not filters:
            return list(state.product_names.keys())

        try:
            import arena_core  # type: ignore
            if not hasattr(arena_core, "build_prefilter_sql"):
                raise ImportError("arena_core native extension not built")
            where_fragment, params = arena_core.build_prefilter_sql(filters)
        except ImportError:
            logger.warning(
                "design_14: arena_core not installed; skipping hard prefilter."
            )
            return list(state.product_names.keys())
        except ValueError as e:
            logger.warning("design_14: prefilter rejected: %s", e)
            return list(state.product_names.keys())

        if not where_fragment:
            return list(state.product_names.keys())

        sql = f"SELECT p.external_id FROM products p WHERE {where_fragment}"
        conn = sqlite3.connect(state.db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            logger.warning("design_14: prefilter SQL failed: %s", e)
            return list(state.product_names.keys())
        finally:
            conn.close()

        candidate_ids = [r[0] for r in rows]
        if not candidate_ids:
            # Filter is too narrow — degrade gracefully so we never return
            # zero results from the system. This matches design 05's behavior.
            logger.debug(
                "design_14: prefilter matched 0 products for filters=%s — "
                "falling back to full catalog", filters,
            )
            return list(state.product_names.keys())
        return candidate_ids

    def _fts5_track(
        self,
        state: "_DomainState",
        query_text: str,
        candidate_ids: list[str],
    ) -> list[tuple[str, float]]:
        try:
            import arena_core  # type: ignore
            if not hasattr(arena_core, "fts5_search"):
                raise ImportError("arena_core native extension not built")
        except ImportError:
            return []

        match_expr = tokenize_for_fts5(query_text)
        if not match_expr:
            return []
        try:
            return arena_core.fts5_search(
                state.db_path,
                match_expr,
                TOP_K_LEXICAL,
                "reviews_fts",
                "product_id",
                candidate_ids if len(candidate_ids) <= 5000 else None,
            )
        except (RuntimeError, ValueError) as e:
            logger.warning("design_14: FTS5 retrieval failed: %s", e)
            return []

    def _vector_track(
        self,
        state: "_DomainState",
        query_text: str,
        candidate_ids: list[str],
    ) -> list[tuple[str, float]]:
        if not state.product_vecs:
            return []
        qvec = retrieval.encode_query(
            query_text,
            model_name=self.embedding_model,
            adapter_path=self.embedding_adapter_path,
        )
        if not qvec:
            return []
        return retrieval.cosine_top(
            qvec, state.product_vecs, top_k=TOP_K_VECTOR, candidate_ids=candidate_ids
        )


# ---------------------------------------------------------------------------
# Per-domain state
# ---------------------------------------------------------------------------
class _DomainState:
    __slots__ = ("db_path", "id_map", "product_names", "product_vecs", "rerank_docs")

    def __init__(
        self,
        db_path: str,
        id_map: dict[str, int],
        product_names: dict[str, str],
        product_vecs: dict[str, list[float]],
        rerank_docs: dict[str, str],
    ):
        self.db_path = db_path
        self.id_map = id_map
        self.product_names = product_names
        self.product_vecs = product_vecs
        self.rerank_docs = rerank_docs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)


def _build_rerank_docs(products: list[dict], reviews: list[dict]) -> dict[str, str]:
    """Build a short text doc per product for the cross-encoder."""
    by_pid: dict[str, list[str]] = {}
    for r in reviews:
        pid = r.get("product_id") or r.get("id") or ""
        text = r.get("text") or r.get("review_text") or ""
        if pid and text:
            by_pid.setdefault(pid, []).append(text)

    docs: dict[str, str] = {}
    for p in products:
        ext_id = p.get("id") or p.get("product_id") or ""
        if not ext_id:
            continue
        parts = [
            p.get("name") or "",
            p.get("brand") or "",
            p.get("category") or "",
        ]
        # Numeric specs (waist_width_mm, lengths_cm, weight_g_per_ski, …)
        # — listwise models with capacity to reason about constraints
        # benefit from seeing exact values, not just attribute scales.
        specs = p.get("specs") or {}
        spec_bits: list[str] = []
        for k, v in sorted(specs.items()):
            if isinstance(v, (int, float, str)):
                spec_bits.append(f"{k}={v}")
            elif isinstance(v, list) and all(isinstance(x, (int, float, str)) for x in v):
                spec_bits.append(f"{k}={','.join(str(x) for x in v)}")
        if spec_bits:
            parts.append(", ".join(spec_bits))
        attrs = p.get("attributes") or {}
        attr_text = ", ".join(
            f"{k}={v}" for k, v in sorted(attrs.items())
            if isinstance(v, (int, float, str))
        )
        if attr_text:
            parts.append(attr_text)
        for r in by_pid.get(ext_id, [])[:2]:
            parts.append(r[:300])
        docs[ext_id] = ". ".join(p for p in parts if p)
    return docs


def _build_explanation(
    filters: list[dict],
    lexical: list[tuple[str, float]],
    vector: list[tuple[str, float]],
    pid: str,
) -> str:
    parts: list[str] = []
    if filters:
        parts.append(
            "matched filters: "
            + ", ".join(f"{f['attribute']} {f['op']} {f['value']}" for f in filters[:3])
        )
    bm = next((s for p, s in lexical if p == pid), None)
    if bm is not None:
        parts.append(f"BM25 score {bm:.2f}")
    cv = next((s for p, s in vector if p == pid), None)
    if cv is not None:
        parts.append(f"vector cos {cv:.2f}")
    parts.append("RRF + cross-encoder rerank")
    return "; ".join(parts) + "."


def _matched_attrs(filters: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in filters:
        out[f["attribute"]] = 1.0
    return out
