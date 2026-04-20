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
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    return path
