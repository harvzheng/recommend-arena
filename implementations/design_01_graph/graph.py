"""Graph construction and querying logic using NetworkX."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _near_miss_score(actual_score: float, desired_polarity: float, *, decay_rate: float = 1.5) -> float:
    """Compute a near-miss tolerant alignment score.

    Instead of simple multiplication (which gives 0 for neutral scores),
    uses exponential decay from the ideal target.

    Args:
        actual_score: The product's attribute score in [-1, 1]
        desired_polarity: What the user wants: +1 (high), -1 (low)
        decay_rate: Controls how quickly score drops off. Higher = stricter.

    Returns:
        Score in [-1, 1] where 1.0 = perfect match.
    """
    # Target is the extreme matching desired polarity
    target = desired_polarity  # +1 or -1

    # Distance from target (0 = perfect, 2 = worst)
    distance = abs(actual_score - target)

    # Exponential decay: score = exp(-decay_rate * distance^2)
    # This gives ~1.0 at distance=0, ~0.4 at distance=0.5, ~0.01 at distance=1.5
    raw = math.exp(-decay_rate * distance * distance)

    # If the product is on the wrong side (e.g., user wants high but product is negative),
    # penalize more
    if actual_score * desired_polarity < 0:
        raw *= 0.3  # Heavy penalty for opposite polarity

    # Scale to [-1, 1]: perfect match -> 1.0, worst -> negative
    return raw * 2.0 - 1.0


class KnowledgeGraph:
    """Property graph that stores products, attributes, reviews, and their relationships."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def clear_domain(self, domain: str) -> None:
        """Remove all nodes belonging to a specific domain."""
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("domain") == domain
        ]
        self.graph.remove_nodes_from(to_remove)

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    def add_product(
        self,
        product_id: str,
        name: str,
        domain: str,
        category: str = "",
        specs: dict | None = None,
        attributes: dict | None = None,
    ) -> None:
        """Add a product node to the graph."""
        self.graph.add_node(
            product_id,
            type="product",
            name=name,
            domain=domain,
            category=category,
            specs=specs or {},
            ground_truth_attributes=attributes or {},
        )
        # Add category edge if we have one
        if category:
            cat_id = f"cat-{domain}-{category}"
            if not self.graph.has_node(cat_id):
                self.graph.add_node(cat_id, type="category", name=category, domain=domain)
            self.graph.add_edge(product_id, cat_id, type="BELONGS_TO", primary=True)

    def add_attribute_node(self, attr_id: str, name: str, domain: str) -> None:
        """Add an attribute node if it doesn't already exist."""
        if not self.graph.has_node(attr_id):
            self.graph.add_node(attr_id, type="attribute", name=name, domain=domain)

    def add_review(
        self,
        review_id: str,
        product_id: str,
        text: str,
        domain: str,
        source: str = "",
        reviewer: str = "",
        rating: float | None = None,
    ) -> None:
        """Add a review node and link it to its product."""
        self.graph.add_node(
            review_id,
            type="review",
            text=text,
            product_id=product_id,
            domain=domain,
            source=source,
            reviewer=reviewer,
            rating=rating,
        )
        if self.graph.has_node(product_id):
            self.graph.add_edge(product_id, review_id, type="REVIEWED_BY", source=source)

    # ------------------------------------------------------------------
    # Edge creation
    # ------------------------------------------------------------------

    def add_mention(
        self,
        review_id: str,
        attr_id: str,
        sentiment: float,
        snippet: str = "",
    ) -> None:
        """Add a MENTIONS edge from a review to an attribute."""
        self.graph.add_edge(
            review_id,
            attr_id,
            type="MENTIONS",
            sentiment=sentiment,
            snippet=snippet,
        )

    def aggregate_has_attribute(self, product_id: str, attr_id: str) -> dict | None:
        """Compute aggregated HAS_ATTRIBUTE from all review mentions.

        Walks: product -> reviews -> attribute mentions, then aggregates.
        """
        if not self.graph.has_node(product_id) or not self.graph.has_node(attr_id):
            return None

        # Find all reviews of this product
        review_ids = [
            target for _, target, data in self.graph.out_edges(product_id, data=True)
            if data.get("type") == "REVIEWED_BY"
        ]

        # Collect mentions of this attribute from those reviews
        sentiments = []
        snippets = []
        for rev_id in review_ids:
            edge_data = self.graph.edges.get((rev_id, attr_id))
            if edge_data and edge_data.get("type") == "MENTIONS":
                sentiments.append(edge_data["sentiment"])
                snippet = edge_data.get("snippet", "")
                if snippet:
                    snippets.append(snippet)

        if not sentiments:
            return None

        score = sum(sentiments) / len(sentiments)
        confidence = min(1.0, len(sentiments) / 5.0)
        return {
            "score": score,
            "confidence": confidence,
            "mention_count": len(sentiments),
            "snippets": snippets[:5],  # Keep top 5 snippets
        }

    def set_has_attribute(
        self,
        product_id: str,
        attr_id: str,
        score: float,
        confidence: float,
        mention_count: int,
        snippets: list[str] | None = None,
    ) -> None:
        """Set or update a HAS_ATTRIBUTE edge."""
        self.graph.add_edge(
            product_id,
            attr_id,
            type="HAS_ATTRIBUTE",
            score=score,
            confidence=confidence,
            mention_count=mention_count,
            snippets=snippets or [],
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_products(self, domain: str) -> list[str]:
        """Get all product IDs for a domain."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "product" and d.get("domain") == domain
        ]

    def get_product_data(self, product_id: str) -> dict:
        """Get node data for a product."""
        return dict(self.graph.nodes[product_id])

    def get_product_category(self, product_id: str) -> str:
        """Get the primary category for a product."""
        for _, target, data in self.graph.out_edges(product_id, data=True):
            if data.get("type") == "BELONGS_TO":
                return self.graph.nodes[target].get("name", "")
        return ""

    def get_has_attribute_edge(self, product_id: str, attr_id: str) -> dict | None:
        """Get the HAS_ATTRIBUTE edge data between a product and attribute."""
        edge_data = self.graph.edges.get((product_id, attr_id))
        if edge_data and edge_data.get("type") == "HAS_ATTRIBUTE":
            return dict(edge_data)
        return None

    def get_product_attributes(self, product_id: str) -> dict[str, dict]:
        """Get all HAS_ATTRIBUTE edges for a product."""
        attrs = {}
        for _, target, data in self.graph.out_edges(product_id, data=True):
            if data.get("type") == "HAS_ATTRIBUTE":
                attr_name = self.graph.nodes[target].get("name", target)
                attrs[attr_name] = dict(data)
        return attrs

    def check_constraint(self, product_id: str, field: str, op: str, value) -> bool:
        """Check if a product meets a spec constraint."""
        product_data = self.graph.nodes.get(product_id, {})
        specs = product_data.get("specs", {})

        spec_value = specs.get(field)
        if spec_value is None or value is None:
            return True  # No data — don't penalize

        # Handle list specs (e.g., lengths_cm is a list of available sizes)
        if isinstance(spec_value, list):
            if op == ">=":
                return any(v >= value for v in spec_value if v is not None)
            elif op == "<=":
                return any(v <= value for v in spec_value if v is not None)
            elif op == "==":
                return value in spec_value
        else:
            if op == ">=":
                return spec_value >= value
            elif op == "<=":
                return spec_value <= value
            elif op == "==":
                return spec_value == value

        return True

    def score_product(
        self,
        product_id: str,
        query_attrs: list[dict],
        domain: str,
    ) -> tuple[float, dict[str, float], list[str]]:
        """Score a product against query attributes.

        Args:
            product_id: The product to score
            query_attrs: List of {attribute, polarity, weight}
            domain: Domain string

        Returns:
            (score, matched_attributes, explanation_snippets)
            score is in [-1, 1] range
        """
        total = 0.0
        weight_sum = 0.0
        matched = {}
        snippets = []

        for qa in query_attrs:
            attr_name = qa["attribute"]
            desired_polarity = qa["polarity"]
            importance = qa.get("weight", 1.0)
            attr_id = f"attr-{domain}-{attr_name}"

            edge = self.get_has_attribute_edge(product_id, attr_id)
            if edge is None:
                # Check ground truth attributes as fallback
                gt_attrs = self.graph.nodes[product_id].get("ground_truth_attributes", {})
                if attr_name in gt_attrs:
                    gt_val = gt_attrs[attr_name]
                    if isinstance(gt_val, (int, float)):
                        # Convert 1-10 scale to -1 to 1
                        normalized = (gt_val - 5.5) / 4.5
                        alignment = _near_miss_score(normalized, desired_polarity)
                        confidence = 0.5  # Lower confidence for ground truth only
                        total += alignment * confidence * importance
                        weight_sum += importance
                        matched[attr_name] = max(0.0, min(1.0, (alignment + 1.0) / 2.0))
                continue

            alignment = _near_miss_score(edge["score"], desired_polarity)
            confidence = edge["confidence"]
            total += alignment * confidence * importance
            weight_sum += importance
            # Store match strength as 0-1
            matched[attr_name] = max(0.0, min(1.0, (alignment + 1.0) / 2.0))

            for snippet in edge.get("snippets", [])[:2]:
                snippets.append(snippet)

        if weight_sum == 0:
            return 0.0, {}, []

        raw_score = total / weight_sum  # [-1, 1]
        return raw_score, matched, snippets

    def get_review_snippets_for_product(
        self, product_id: str, attr_name: str, domain: str
    ) -> list[str]:
        """Get review snippets that mention a specific attribute for a product."""
        attr_id = f"attr-{domain}-{attr_name}"
        snippets = []

        # Find all reviews of this product
        for _, target, data in self.graph.out_edges(product_id, data=True):
            if data.get("type") == "REVIEWED_BY":
                review_id = target
                edge_data = self.graph.edges.get((review_id, attr_id))
                if edge_data and edge_data.get("type") == "MENTIONS":
                    snippet = edge_data.get("snippet", "")
                    if snippet:
                        snippets.append(snippet)

        return snippets
