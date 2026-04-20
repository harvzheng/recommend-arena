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
    """Extract the first JSON object from model output."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

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
    """Parse raw model output into a StudentJudgment. Fallback on failure."""
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
        logger.warning("Failed to parse student output for %s: %s", product_name, e)
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
    """Wraps a loaded student model for inference. Model/tokenizer are injected."""

    def __init__(self, model, tokenizer, max_seq_length: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def _build_prompt(self, query: str, product_ctx: ProductContext) -> str:
        return (
            f"### Instruction:\n{STUDENT_INSTRUCTION}\n\n"
            f"### Input:\n"
            f"Rate this product for the query: {query}\n\n"
            f"Product: {product_ctx.product_name}\n"
            f"{product_ctx.context_text}\n\n"
            f"### Response:\n"
        )

    def infer(self, query: str, product_ctx: ProductContext) -> StudentJudgment:
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
