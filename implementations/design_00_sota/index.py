"""In-memory product index with embeddings and structured attributes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Pre-defined attribute schemas per domain (instead of inferring from first product)
DOMAIN_ATTRIBUTE_SCHEMAS: dict[str, list[str]] = {
    "ski": [
        "stiffness", "damp", "edge_grip", "stability", "powder_float",
        "playfulness", "responsiveness", "turn_initiation", "versatility",
        "vibration_absorption",
    ],
    "running_shoe": [
        "cushioning", "responsiveness", "stability", "grip", "breathability",
        "durability", "weight", "comfort", "support", "flexibility",
    ],
}


@dataclass
class ProductRecord:
    """All data about a single product for retrieval and scoring."""
    product_id: str
    product_name: str
    domain: str
    category: str
    specs: dict                          # raw spec dict from product data
    ground_truth_attributes: dict        # 1-10 scale from product data
    review_attributes: dict              # ABSA-extracted: {attr: {score, confidence, snippets}}
    embedding: list[float] | None = None # dense vector for semantic search
    review_text_combined: str = ""       # concatenated reviews for embedding


@dataclass
class ProductIndex:
    """In-memory hybrid index: dense embeddings + structured attributes."""

    products: dict[str, ProductRecord] = field(default_factory=dict)
    domain_attributes: dict[str, list[str]] = field(default_factory=dict)
    domain_categories: dict[str, set[str]] = field(default_factory=dict)
    domain_spec_fields: dict[str, list[str]] = field(default_factory=dict)

    def add_product(self, record: ProductRecord) -> None:
        self.products[record.product_id] = record
        domain = record.domain
        if domain not in self.domain_attributes:
            # Use pre-defined schema if available, fall back to inference
            if domain in DOMAIN_ATTRIBUTE_SCHEMAS:
                self.domain_attributes[domain] = list(DOMAIN_ATTRIBUTE_SCHEMAS[domain])
            else:
                attr_names = [k for k, v in record.ground_truth_attributes.items()
                             if isinstance(v, (int, float))]
                self.domain_attributes[domain] = attr_names
        if domain not in self.domain_categories:
            self.domain_categories[domain] = set()
        self.domain_categories[domain].add(record.category)
        if domain not in self.domain_spec_fields:
            self.domain_spec_fields[domain] = list(record.specs.keys())

    def get_domain_products(self, domain: str) -> list[ProductRecord]:
        return [p for p in self.products.values() if p.domain == domain]

    def get_attribute_names(self, domain: str) -> list[str]:
        return self.domain_attributes.get(domain, [])

    def get_spec_fields(self, domain: str) -> list[str]:
        return self.domain_spec_fields.get(domain, [])

    def get_categories(self, domain: str) -> list[str]:
        return sorted(self.domain_categories.get(domain, set()))

    def clear_domain(self, domain: str) -> None:
        self.products = {pid: p for pid, p in self.products.items()
                         if p.domain != domain}
        self.domain_attributes.pop(domain, None)
        self.domain_categories.pop(domain, None)
        self.domain_spec_fields.pop(domain, None)

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def semantic_search(self, query_embedding: list[float], domain: str,
                        top_k: int) -> list[tuple[ProductRecord, float]]:
        """Return top_k products by cosine similarity to query embedding."""
        candidates = self.get_domain_products(domain)
        scored = []
        for p in candidates:
            if p.embedding is None:
                continue
            sim = self.cosine_similarity(query_embedding, p.embedding)
            scored.append((p, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
