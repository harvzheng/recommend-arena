"""Generative listwise reranker — Qwen3-1.7B by default.

Mirrors the design-13 (Opus) listwise approach but runs locally via
transformers + MPS. The model reads the query and all N candidates in
one prompt, then emits an ordered permutation like `[3] > [7] > [1] >
...` which we parse back into a ranking.

Why listwise over pointwise:
  - Pointwise rerankers (bge-reranker, ms-marco) score each (query, doc)
    pair independently. They cannot reason "candidate A is more
    confidence-inspiring than candidate B given the same query."
  - Listwise sees all candidates at once. Wins on subjective / vague
    queries ("ice coast", "confidence-inspiring") where the relative
    judgment matters more than absolute relevance.

Lazy-imports torch/transformers; pass-through if unavailable.

The candidate formatter is deterministic (stable iteration order, sorted
attribute keys) so repeated calls with the same candidate set hit the
KV cache.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LISTWISE_MODEL = os.environ.get(
    "RECOMMEND_LISTWISE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
)

_listwise_cache: dict[str, Any] = {}


def _load_model(model_name: str, adapter_path: str | None = None):
    cache_key = f"{model_name}::{adapter_path or ''}"
    if cache_key in _listwise_cache:
        return _listwise_cache[cache_key]

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        logger.warning(
            "design_14: transformers/torch not available; "
            "listwise reranker disabled (pass-through)."
        )
        _listwise_cache[cache_key] = None
        return None

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("design_14: loading listwise reranker %s on %s …", model_name, device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    if adapter_path:
        try:
            from peft import PeftModel  # type: ignore
            logger.info("design_14: applying listwise LoRA adapter %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
        except ImportError:
            logger.warning("peft not installed; ignoring listwise adapter_path")
    model = model.to(device).eval()
    _listwise_cache[cache_key] = (tok, model, device)
    return _listwise_cache[cache_key]


SYSTEM_PROMPT = (
    "You are an expert product-ranking model. Given a user query and a "
    "numbered list of candidate products with their attributes, return a "
    "permutation that ranks the candidates from most to least relevant. "
    "Reason carefully about: numeric constraints (e.g. 'over 100mm waist'), "
    "negation ('not titanal', 'not playful'), and trade-offs (e.g. 'forgiving "
    "but not a beginner ski'). The OUTPUT must be a single line in the format "
    "`[N] > [N] > [N] > ...` listing every candidate number exactly once. "
    "Do not add any other text."
)


MAX_DOC_CHARS = int(os.environ.get("RECOMMEND_LISTWISE_MAX_DOC_CHARS", "800"))


def format_candidate(idx: int, pid: str, doc: str) -> str:
    """Deterministic per-candidate block. Index is 1-based to match the
    [1] > [2] convention used in RankGPT-style prompts.

    Doc is truncated to MAX_DOC_CHARS to keep prompts MPS-tractable.
    Training and inference must use the same limit so the model sees
    the same surface form it was trained on.
    """
    doc = doc.strip().replace("\n", " ")
    if len(doc) > MAX_DOC_CHARS:
        doc = doc[:MAX_DOC_CHARS] + "…"
    return f"[{idx}] {pid}: {doc}"


def build_prompt(query: str, candidates: list[tuple[str, str]]) -> str:
    """Render the full ranking prompt deterministically."""
    blocks = [format_candidate(i + 1, pid, doc) for i, (pid, doc) in enumerate(candidates)]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Query: {query}\n\n"
        f"Candidates:\n" + "\n".join(blocks) + "\n\n"
        f"Ranking: "
    )


_RANK_RE = re.compile(r"\[(\d+)\]")


def parse_ranking(text: str, n: int) -> list[int]:
    """Pull the [k] tokens out of the model's output, in order.

    Returns 0-based indices. Defensive: if parsing fails or the model
    returns fewer than n picks, append the missing indices in original
    order so we never lose a candidate.
    """
    nums: list[int] = []
    seen: set[int] = set()
    for m in _RANK_RE.finditer(text):
        v = int(m.group(1)) - 1
        if 0 <= v < n and v not in seen:
            seen.add(v)
            nums.append(v)
        if len(nums) == n:
            break
    # Append any missing indices in original order
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_k: int = 10,
    model_name: str = DEFAULT_LISTWISE_MODEL,
    adapter_path: str | None = None,
    max_new_tokens: int = 128,
) -> list[tuple[str, float]]:
    """Listwise rerank. Returns (id, score) where score is 1/(rank+1).

    Pass-through (preserves input order) when the model can't be loaded.
    """
    if not candidates:
        return []

    handle = _load_model(model_name, adapter_path=adapter_path)
    if handle is None:
        return [(pid, 1.0 / (i + 1)) for i, (pid, _) in enumerate(candidates)]

    tok, model, device = handle
    n = len(candidates)
    prompt = build_prompt(query, candidates)

    # Use the chat template if the tokenizer has one — instruct-tuned models
    # like Qwen3-4B-Instruct ship with a chat template and behave noticeably
    # better when input is rendered as a chat turn vs. a raw prompt.
    if getattr(tok, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt_text = prompt

    import torch  # type: ignore
    inputs = tok(
        prompt_text, return_tensors="pt", truncation=True, max_length=8192,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Reasoning-tuned models may wrap their CoT in <think>...</think> before
    # emitting the answer. Drop that wrapper so the rank parser sees the
    # final permutation cleanly.
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    logger.debug("listwise raw output: %s", completion[:200])
    order = parse_ranking(completion, n)

    scored = [(candidates[idx][0], 1.0 / (rank + 1)) for rank, idx in enumerate(order)]
    return scored[:top_k]
