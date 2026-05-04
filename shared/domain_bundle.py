"""Per-domain artifact bundle — the unit the framework produces and consumes.

A bundle is a self-contained directory holding everything needed to run a
recommender on one product domain: SQLite + FTS5 index, embedding (vanilla
or fine-tuned), optional reranker LoRA, filter schema, eval set, and a
manifest pinning the versions of every component.

Layout (matches the task description):

    artifacts/<domain>/
      manifest.json
      fts5.db                    # SQLite + FTS5 index
      products.jsonl             # canonical product list (for re-builds)
      reviews.jsonl              # canonical review list
      schema.py                  # Pydantic filter schema (optional, slice 14.1+)
      embedding/                 # vanilla or fine-tuned (optional)
        adapter_model.safetensors  (LoRA, ~50-150MB)
        OR full weights (~1.2GB)
        OR a pointer file `model_id.txt` for the off-the-shelf case
      reranker_lora/             # fine-tuned reranker (optional)
        adapter_model.safetensors  (~30MB)
      eval/
        queries.jsonl            # ground-truth + synthetic eval queries
        baseline_scores.json     # NDCG of vanilla baselines for the gate

The CLI `arena_new.py` produces this directory. The runtime
`LocalHybridRecommender.from_bundle(path)` loads it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Bumped whenever the on-disk format changes in a way that invalidates
# older bundles. The runtime should refuse to load a bundle whose
# manifest_version is higher than this.
MANIFEST_VERSION = 1


@dataclass
class EmbeddingArtifact:
    """Where the embedding model lives, and what kind it is."""

    # 'off_the_shelf' | 'lora' | 'full_finetune'
    kind: str
    # For off_the_shelf: the HF model id. For lora/full_finetune: the
    # base model id the adapter was trained against.
    base_model: str
    # Relative path from the bundle root to the adapter dir or full
    # weights dir. None for off_the_shelf (we just instantiate from
    # base_model directly).
    adapter_path: Optional[str] = None
    # Embedding dimension. Recorded so the FTS5 + vector tracks can
    # sanity-check on load.
    dim: int = 1024


@dataclass
class RerankerArtifact:
    """Cross-encoder reranker (optional)."""

    kind: str  # 'off_the_shelf' | 'lora'
    base_model: str
    adapter_path: Optional[str] = None


@dataclass
class EvalSummary:
    """Per-bundle baselines used by the eval gate."""

    # NDCG@5 of #5 (SQL+FTS5) and #11 (fine-tuned embeddings) when each is
    # run against THIS bundle's eval set. Fine-tuned components must beat
    # both before we'll ship.
    fts5_baseline_ndcg5: Optional[float] = None
    vanilla_embedding_baseline_ndcg5: Optional[float] = None
    # The full design-14 pipeline score on the eval set.
    full_pipeline_ndcg5: Optional[float] = None
    # The arena-wide threshold the full pipeline must beat before we ship
    # (current arena #1 = 0.527; spec sets this to 0.55).
    arena_threshold_ndcg5: float = 0.55


@dataclass
class Manifest:
    """The on-disk descriptor of a bundle.

    Everything the runtime needs to know about WHAT this bundle is goes
    here. The bundle's data files (FTS5 db, embeddings, etc) live alongside.
    """

    domain: str
    manifest_version: int = MANIFEST_VERSION
    arena_core_min_version: str = "0.1.0"
    n_products: int = 0
    n_reviews: int = 0
    embedding: Optional[EmbeddingArtifact] = None
    reranker: Optional[RerankerArtifact] = None
    eval: EvalSummary = field(default_factory=EvalSummary)
    # Free-form metadata: training run info, source dataset URL, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        # Convert nested dicts back into their typed dataclasses.
        emb = d.get("embedding")
        rer = d.get("reranker")
        ev = d.get("eval") or {}
        return cls(
            domain=d["domain"],
            manifest_version=d.get("manifest_version", MANIFEST_VERSION),
            arena_core_min_version=d.get("arena_core_min_version", "0.1.0"),
            n_products=d.get("n_products", 0),
            n_reviews=d.get("n_reviews", 0),
            embedding=EmbeddingArtifact(**emb) if emb else None,
            reranker=RerankerArtifact(**rer) if rer else None,
            eval=EvalSummary(**ev),
            metadata=d.get("metadata") or {},
        )


@dataclass
class BundlePaths:
    """All the relative paths inside a bundle directory."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def fts5_db(self) -> Path:
        return self.root / "fts5.db"

    @property
    def products_jsonl(self) -> Path:
        return self.root / "products.jsonl"

    @property
    def reviews_jsonl(self) -> Path:
        return self.root / "reviews.jsonl"

    @property
    def schema_py(self) -> Path:
        return self.root / "schema.py"

    @property
    def embedding_dir(self) -> Path:
        return self.root / "embedding"

    @property
    def reranker_dir(self) -> Path:
        return self.root / "reranker_lora"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def eval_queries_jsonl(self) -> Path:
        return self.eval_dir / "queries.jsonl"

    @property
    def baseline_scores_json(self) -> Path:
        return self.eval_dir / "baseline_scores.json"


class Bundle:
    """In-memory handle to an on-disk bundle directory.

    Construct via `Bundle.create(path, domain)` for a fresh bundle
    or `Bundle.load(path)` for an existing one.
    """

    def __init__(self, paths: BundlePaths, manifest: Manifest):
        self.paths = paths
        self.manifest = manifest

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, domain: str) -> "Bundle":
        """Create a new empty bundle directory at *root*."""
        paths = BundlePaths(root=Path(root))
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.eval_dir.mkdir(parents=True, exist_ok=True)
        manifest = Manifest(domain=domain)
        bundle = cls(paths, manifest)
        bundle.save_manifest()
        return bundle

    @classmethod
    def load(cls, root: str | Path) -> "Bundle":
        """Load an existing bundle. Raises if the manifest is missing."""
        paths = BundlePaths(root=Path(root))
        if not paths.manifest_path.is_file():
            raise FileNotFoundError(
                f"no manifest.json found in {paths.root!s}; "
                f"is this a bundle directory?"
            )
        with paths.manifest_path.open() as f:
            manifest = Manifest.from_dict(json.load(f))
        if manifest.manifest_version > MANIFEST_VERSION:
            raise RuntimeError(
                f"bundle {paths.root!s} has manifest_version "
                f"{manifest.manifest_version}, but this code only "
                f"understands up to {MANIFEST_VERSION}. Upgrade arena_core."
            )
        return cls(paths, manifest)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_manifest(self) -> None:
        with self.paths.manifest_path.open("w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

    def write_products(self, products: list[dict]) -> None:
        with self.paths.products_jsonl.open("w") as f:
            for p in products:
                f.write(json.dumps(p, sort_keys=True))
                f.write("\n")
        self.manifest.n_products = len(products)

    def write_reviews(self, reviews: list[dict]) -> None:
        with self.paths.reviews_jsonl.open("w") as f:
            for r in reviews:
                f.write(json.dumps(r, sort_keys=True))
                f.write("\n")
        self.manifest.n_reviews = len(reviews)

    def read_products(self) -> list[dict]:
        return _read_jsonl(self.paths.products_jsonl)

    def read_reviews(self) -> list[dict]:
        return _read_jsonl(self.paths.reviews_jsonl)

    def write_eval_queries(self, queries: list[dict]) -> None:
        self.paths.eval_dir.mkdir(parents=True, exist_ok=True)
        with self.paths.eval_queries_jsonl.open("w") as f:
            for q in queries:
                f.write(json.dumps(q, sort_keys=True))
                f.write("\n")

    def read_eval_queries(self) -> list[dict]:
        if not self.paths.eval_queries_jsonl.is_file():
            return []
        return _read_jsonl(self.paths.eval_queries_jsonl)

    def write_baseline_scores(self, scores: dict[str, Any]) -> None:
        self.paths.eval_dir.mkdir(parents=True, exist_ok=True)
        with self.paths.baseline_scores_json.open("w") as f:
            json.dump(scores, f, indent=2, sort_keys=True)
            f.write("\n")

    def read_baseline_scores(self) -> dict[str, Any]:
        if not self.paths.baseline_scores_json.is_file():
            return {}
        with self.paths.baseline_scores_json.open() as f:
            return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
