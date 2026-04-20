"""Embedding and indexing logic for Design #2.

Handles:
- Review chunking into opinion-bearing passages
- Embedding via the shared LLM provider
- ChromaDB collection management and upsert
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

def segment_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex splitter."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


# ---------------------------------------------------------------------------
# Opinion filtering
# ---------------------------------------------------------------------------

SKIP_PATTERNS = [
    r"\b(ship|deliver|packag|return|refund|box|arrived|tracking)\b",
    r"\b(customer service|support ticket|warranty claim)\b",
    r"\b(unbox|unwrap)\b",
]

# Simple adjective/descriptor signal words that indicate an opinion
OPINION_SIGNALS = re.compile(
    r"\b("
    r"stiff|soft|flexible|forgiving|rigid|firm|"
    r"light|lightweight|heavy|hefty|"
    r"stable|steady|planted|wobbly|unstable|"
    r"damp|smooth|chattery|vibrat\w+|composed|"
    r"fast|quick|slow|sluggish|responsive|snappy|"
    r"grip|edge|icy|slip|traction|bite|"
    r"playful|fun|lively|boring|dead|"
    r"precise|accurate|sloppy|"
    r"float|powder|rocker|camber|"
    r"great|excellent|good|bad|terrible|amazing|incredible|perfect|"
    r"best|worst|love|hate|"
    r"comfortable|uncomfortable|cushy|harsh|"
    r"versatile|one-dimensional|"
    r"forgiving|demanding|punishing|"
    r"poppy|energetic|springy|"
    r"wide|narrow|"
    r"long|short|"
    r"easy|difficult|hard|"
    r"powerful|weak"
    r")\b",
    re.IGNORECASE,
)


def is_opinion_bearing(text: str) -> bool:
    """Return True if the text appears to contain opinion/attribute content."""
    # Skip logistics-only sentences
    if any(re.search(p, text, re.IGNORECASE) for p in SKIP_PATTERNS):
        # Only skip if there are no opinion signals
        if not OPINION_SIGNALS.search(text):
            return False
    # Keep if it contains opinion/descriptor words
    return bool(OPINION_SIGNALS.search(text))


# ---------------------------------------------------------------------------
# Review chunking
# ---------------------------------------------------------------------------

def chunk_review(
    review: dict,
    window_size: int = 3,
    step: int = 2,
) -> list[dict]:
    """Chunk a single review into opinion-bearing passages.

    Uses a sliding window of `window_size` sentences with `step` sentence stride.
    Filters out non-opinion passages.

    Args:
        review: Dict with keys: product_id, text, reviewer, rating
        window_size: Number of sentences per chunk
        step: Stride between chunks

    Returns:
        List of passage dicts with keys: product_id, text, review_id
    """
    text = review.get("text", review.get("review_text", ""))
    product_id = review.get("product_id", "")
    review_id = review.get("review_id", review.get("reviewer", str(uuid.uuid4())))

    sentences = segment_sentences(text)
    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), step):
        chunk_text = " ".join(sentences[i : i + window_size])
        if is_opinion_bearing(chunk_text):
            chunks.append(
                {
                    "product_id": product_id,
                    "text": chunk_text,
                    "review_id": review_id,
                }
            )

    # If sliding window produced nothing but the full review has opinions, use the full text
    if not chunks and is_opinion_bearing(text):
        chunks.append(
            {
                "product_id": product_id,
                "text": text,
                "review_id": review_id,
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_texts(provider: LLMProvider, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the shared LLM provider.

    Processes sequentially since the provider handles one text at a time.
    """
    vectors = []
    for text in texts:
        vec = provider.embed(text)
        vectors.append(vec)
    return vectors


# ---------------------------------------------------------------------------
# ChromaDB index management
# ---------------------------------------------------------------------------

def build_index(
    client: chromadb.ClientAPI,
    provider: LLMProvider,
    products: list[dict],
    reviews: list[dict],
    domain: str,
) -> tuple[chromadb.Collection, dict[str, str]]:
    """Build the ChromaDB passage index for a domain.

    Returns:
        Tuple of (collection, product_names_map)
    """
    # Delete existing collection for idempotency
    try:
        client.delete_collection(f"passages_{domain}")
    except Exception:
        pass

    collection = client.create_collection(
        name=f"passages_{domain}",
        metadata={"hnsw:space": "cosine"},
    )

    # Build product name lookup
    product_names: dict[str, str] = {}
    for p in products:
        pid = p.get("product_id", p.get("id", ""))
        pname = p.get("product_name", p.get("name", pid))
        product_names[pid] = pname

    # Chunk all reviews into passages
    all_passages: list[dict] = []
    for review in reviews:
        chunks = chunk_review(review)
        all_passages.extend(chunks)

    if not all_passages:
        logger.warning("No passages generated for domain %s", domain)
        return collection, product_names

    logger.info(
        "Domain %s: %d reviews -> %d passages", domain, len(reviews), len(all_passages)
    )

    # Embed all passages
    texts = [p["text"] for p in all_passages]
    vectors = embed_texts(provider, texts)

    # Upsert in batches
    batch_size = 256
    for i in range(0, len(all_passages), batch_size):
        j = min(i + batch_size, len(all_passages))
        batch_ids = [str(uuid.uuid4()) for _ in range(i, j)]
        batch_docs = texts[i:j]
        batch_embeds = vectors[i:j]
        batch_metas = [
            {
                "product_id": p["product_id"],
                "review_id": p["review_id"],
                "domain": domain,
            }
            for p in all_passages[i:j]
        ]
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeds,
            documents=batch_docs,
            metadatas=batch_metas,
        )

    logger.info("Indexed %d passages into collection passages_%s", len(all_passages), domain)
    return collection, product_names
