"""Design #12: Distilled LLM Ranker.

Uses a teacher LLM to label query-product pairs at training time,
then a LoRA-tuned Qwen 2.5 student for local inference at query time.
"""

from __future__ import annotations

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
        if self._inference is None:
            self._inference = self._load_student()
        return self._inference

    def _load_student(self) -> StudentInference:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading student model: %s + %s", self.base_model_name, self.adapter_path)

        # Resolve device — MPS needs float32 (float16 has limited op support)
        device = self.device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        dtype = torch.float32 if device == "mps" else torch.float16

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype,
            device_map={"": device} if device != "cpu" else None,
        )

        if self.adapter_path.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(self.adapter_path))
        else:
            logger.warning(
                "Adapter not found at %s; using base model (untrained).",
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

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
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
        logger.info("Ingested %d products for domain '%s'", len(products), domain)

    def query(
        self, query_text: str, domain: str, top_k: int = 10,
    ) -> list[RecommendationResult]:
        candidates = [
            ctx for ctx in self.contexts.values()
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
            results.append(RecommendationResult(
                product_id=j.product_id,
                product_name=j.product_name,
                score=round(j.score, 4),
                explanation=j.explanation,
                matched_attributes=j.matched_attributes,
            ))
        return results
