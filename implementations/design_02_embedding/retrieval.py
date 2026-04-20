"""Query encoding, retrieval, contrastive scoring, and attribute extraction.

Implements the core query pipeline for Design #2:
1. Encode positive + contrastive negative query
2. Retrieve top passages from ChromaDB
3. Contrastive scoring per passage
4. Aggregate by product (sum-of-top-K)
5. Normalize scores to 0-1
6. Extract matched attributes from evidence passages
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import chromadb
    from shared.llm_provider import LLMProvider


# ---------------------------------------------------------------------------
# Contrastive negative query generation
# ---------------------------------------------------------------------------

NEGATION_MAP = {
    "stiff": "soft forgiving",
    "soft": "stiff rigid",
    "light": "heavy bulky",
    "lightweight": "heavy bulky",
    "heavy": "light lightweight",
    "fast": "slow sluggish",
    "slow": "fast quick",
    "stable": "unstable wobbly",
    "damp": "chattery vibrating harsh",
    "responsive": "sluggish dead unresponsive",
    "grippy": "slippery loose",
    "playful": "boring dead serious",
    "forgiving": "demanding punishing harsh",
    "versatile": "one-dimensional limited",
    "wide": "narrow skinny",
    "narrow": "wide bulky",
    "precise": "sloppy imprecise",
    "powerful": "weak wimpy",
    "comfortable": "uncomfortable harsh",
    "smooth": "rough chattery",
    "aggressive": "mellow gentle tame",
    "fun": "boring dull serious",
    "float": "sink",
    "powder": "hardpack groomed",
    "carving": "smearing sliding",
    "edge": "flat sliding",
}


def generate_neg_query(query: str) -> str:
    """Generate a contrastive negative query by flipping key attribute terms.

    Non-attribute words are kept the same to maintain domain context.
    """
    words = query.lower().split()
    neg_words = []
    for w in words:
        # Strip punctuation for matching
        clean = re.sub(r"[^a-z]", "", w)
        if clean in NEGATION_MAP:
            neg_words.append(NEGATION_MAP[clean])
        else:
            neg_words.append(w)
    return " ".join(neg_words)


# ---------------------------------------------------------------------------
# Attribute extraction patterns
# ---------------------------------------------------------------------------

ATTRIBUTE_PATTERNS = {
    "stiffness": r"\b(stiff|rigid|firm|soft|flexible|forgiving|flex)\b",
    "weight": r"\b(light|lightweight|heavy|hefty|featherweight|weight)\b",
    "stability": r"\b(stable|steady|planted|wobbly|unstable|locked.in|stability)\b",
    "dampness": r"\b(damp|smooth|chattery|vibrat\w+|composed|dampness|dampening)\b",
    "speed": r"\b(fast|quick|slow|sluggish|responsive|snappy|speed)\b",
    "grip": r"\b(grip|edge.hold|icy|slip|traction|bite|edge)\b",
    "playfulness": r"\b(playful|fun|lively|boring|dead|energetic|poppy)\b",
    "versatility": r"\b(versatile|all.mountain|quiver|one.dimensional|limited)\b",
    "float": r"\b(float|powder|sink|surf)\b",
    "forgiveness": r"\b(forgiving|forgiveness|punishing|demanding|harsh|friendly)\b",
    "precision": r"\b(precise|precision|accurate|sloppy|imprecise)\b",
    "comfort": r"\b(comfortable|uncomfortable|cushy|harsh|smooth)\b",
    "turn_quality": r"\b(carv\w+|turn\w*|arc|radius|initiat\w+)\b",
}


def extract_attributes(
    evidence: list[str],
    query_vec: list[float],
    provider: LLMProvider,
) -> dict[str, float]:
    """Extract matched attributes from evidence passages.

    Uses regex keyword matching weighted by passage-query embedding similarity.
    Returns attribute name -> relevance score (0-1).
    """
    query_arr = np.array(query_vec, dtype=np.float32)
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-10)

    attr_scores: dict[str, list[float]] = defaultdict(list)

    for passage in evidence:
        p_vec = np.array(provider.embed(passage), dtype=np.float32)
        p_norm = p_vec / (np.linalg.norm(p_vec) + 1e-10)
        sim = float(np.dot(p_norm, query_norm))

        for attr_name, pattern in ATTRIBUTE_PATTERNS.items():
            if re.search(pattern, passage, re.IGNORECASE):
                attr_scores[attr_name].append(sim)

    # Average similarity for each matched attribute, clamped to [0, 1]
    return {
        attr: round(min(1.0, max(0.0, sum(scores) / len(scores))), 4)
        for attr, scores in attr_scores.items()
    }


# ---------------------------------------------------------------------------
# Core retrieval and scoring
# ---------------------------------------------------------------------------

def retrieve_and_score(
    collection: chromadb.Collection,
    provider: LLMProvider,
    query_text: str,
    domain: str,
    product_names: dict[str, str],
    top_k: int = 10,
    n_retrieve: int = 200,
    top_k_per_product: int = 5,
    n_evidence: int = 5,
) -> list[dict]:
    """Full retrieval pipeline: embed query, retrieve passages, score products.

    Args:
        collection: ChromaDB collection with passage embeddings
        provider: LLM provider for embedding
        query_text: Natural language query
        domain: Product domain
        product_names: Mapping of product_id -> product_name
        top_k: Number of products to return
        n_retrieve: Number of passages to retrieve from the index
        top_k_per_product: Number of top passage scores to sum per product
        n_evidence: Number of evidence passages to keep per product

    Returns:
        List of result dicts with keys:
            product_id, product_name, score, evidence, matched_attributes, explanation
    """
    # Step 1: Encode positive and negative queries
    neg_query = generate_neg_query(query_text)
    pos_vec = provider.embed(query_text)
    neg_vec = provider.embed(neg_query)

    pos_arr = np.array(pos_vec, dtype=np.float32)
    neg_arr = np.array(neg_vec, dtype=np.float32)

    # Normalize
    pos_arr = pos_arr / (np.linalg.norm(pos_arr) + 1e-10)
    neg_arr = neg_arr / (np.linalg.norm(neg_arr) + 1e-10)

    # Step 2: Retrieve top passages by positive query
    # Limit n_results to what the collection actually has
    count = collection.count()
    actual_n = min(n_retrieve, count)
    if actual_n == 0:
        return []

    results = collection.query(
        query_embeddings=[pos_vec],
        n_results=actual_n,
        where={"domain": domain},
        include=["embeddings", "documents", "metadatas"],
    )

    docs = results["documents"][0]
    embeds = np.array(results["embeddings"][0], dtype=np.float32)
    metas = results["metadatas"][0]

    if len(docs) == 0:
        return []

    # Normalize passage embeddings
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    embeds_normed = embeds / norms

    # Step 3: Contrastive scoring per passage
    pos_scores = embeds_normed @ pos_arr
    neg_scores = embeds_normed @ neg_arr
    contrastive = pos_scores - neg_scores  # range roughly [-2, 2]

    # Step 4: Aggregate by product (sum-of-top-K)
    product_scores: dict[str, list[float]] = defaultdict(list)
    product_evidence: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for i, meta in enumerate(metas):
        pid = meta["product_id"]
        score = float(contrastive[i])
        product_scores[pid].append(score)
        product_evidence[pid].append((score, docs[i]))

    scored = []
    for pid, scores in product_scores.items():
        top = sorted(scores, reverse=True)[:top_k_per_product]
        raw_score = sum(top)

        # Sort evidence by score descending, keep top N
        sorted_ev = sorted(product_evidence[pid], key=lambda x: x[0], reverse=True)
        evidence_texts = [text for _, text in sorted_ev[:n_evidence]]

        scored.append(
            {
                "product_id": pid,
                "raw_score": raw_score,
                "evidence": evidence_texts,
            }
        )

    if not scored:
        return []

    # Step 5: Normalize scores to 0-1 (min-max)
    raw_scores = [s["raw_score"] for s in scored]
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    score_range = max_s - min_s if max_s > min_s else 1.0

    for s in scored:
        s["score"] = (s["raw_score"] - min_s) / score_range

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Step 6: Build final results with attribute extraction (top_k only)
    output = []
    for item in scored[:top_k]:
        pid = item["product_id"]
        evidence = item["evidence"][:3]  # use top 3 for attribute extraction

        attrs = extract_attributes(evidence, pos_vec, provider)

        product_name = product_names.get(pid, pid)
        n_attrs = len(attrs)
        n_passages = len(item["evidence"])

        output.append(
            {
                "product_id": pid,
                "product_name": product_name,
                "score": round(item["score"], 4),
                "evidence": item["evidence"],
                "matched_attributes": attrs,
                "explanation": (
                    f"Matched on {n_attrs} attribute(s) from {n_passages} review passage(s). "
                    f"Top evidence: \"{evidence[0][:120]}...\""
                    if evidence
                    else f"Matched from {n_passages} review passage(s)."
                ),
            }
        )

    return output
