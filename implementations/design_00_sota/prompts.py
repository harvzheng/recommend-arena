"""All LLM prompt templates for the SOTA recommender."""

ABSA_EXTRACTION_PROMPT = """\
You are an expert product reviewer analyst. Given a product and its reviews, extract structured attribute assessments.

Product: {product_name} ({category})
Domain: {domain}
Known attributes for this domain: {attribute_names}
Product specs: {specs_json}

Reviews:
{reviews_text}

For each known attribute, analyze the reviews and provide:
- score: float from 0.0 to 10.0 based on reviewer consensus (null if not mentioned)
- confidence: float from 0.0 to 1.0 (how many reviews agree / how clear the signal is)
- snippets: list of 1-3 short direct quotes that support the score

Also extract any additional attributes reviewers mention that aren't in the known list.

Respond with JSON only:
{{
  "attributes": {{
    "attribute_name": {{
      "score": 7.5,
      "confidence": 0.8,
      "snippets": ["quote from review"]
    }}
  }},
  "additional_attributes": {{
    "new_attr_name": {{
      "score": 6.0,
      "confidence": 0.5,
      "snippets": ["quote"]
    }}
  }}
}}"""

QUERY_PARSE_PROMPT = """\
You are a search query analyzer for a {domain} recommendation system.

Available attributes in this domain (1-10 scale): {attribute_names}
Available spec fields: {spec_fields}
Available categories: {categories}

Parse this natural language query into structured search parameters.

Query: "{query_text}"

Instructions:
- Extract desired attributes with importance weights (0.0-1.0)
- Extract spec constraints as field/operator/value triples
- Detect negations ("NOT playful", "not too stiff") as negative attributes
- Expand vague/colloquial terms into concrete attributes
- Identify target categories if mentioned or implied
- For attributes, use ONLY names from the available attributes list above
- When the query mentions vibes or feelings, translate them into multiple concrete attributes with appropriate weights

Here are examples of correct parsing:

Example 1 - Colloquial/vague query:
Query: "Looking for an ice coast ripper that feels alive underfoot"
{{
  "desired_attributes": [
    {{"name": "edge_grip", "weight": 0.95, "direction": "high"}},
    {{"name": "damp", "weight": 0.8, "direction": "high"}},
    {{"name": "stability", "weight": 0.7, "direction": "high"}},
    {{"name": "playfulness", "weight": 0.85, "direction": "high"}},
    {{"name": "responsiveness", "weight": 0.85, "direction": "high"}}
  ],
  "negative_attributes": [],
  "spec_constraints": [],
  "categories": [],
  "expanded_terms": ["edge_grip", "damp", "stability", "playfulness", "responsiveness", "hardpack"],
  "query_embedding_text": "ski with excellent edge grip and dampness for hardpack ice conditions that also feels playful and responsive underfoot"
}}

Example 2 - Negation handling:
Query: "Stable charger for groomers, NOT playful, 180cm or longer"
{{
  "desired_attributes": [
    {{"name": "stability", "weight": 0.95, "direction": "high"}},
    {{"name": "stiffness", "weight": 0.9, "direction": "high"}},
    {{"name": "damp", "weight": 0.85, "direction": "high"}},
    {{"name": "edge_grip", "weight": 0.8, "direction": "high"}}
  ],
  "negative_attributes": [
    {{"name": "playfulness", "weight": 0.8, "direction": "low"}}
  ],
  "spec_constraints": [
    {{"field": "lengths_cm", "op": ">=", "value": 180}}
  ],
  "categories": ["frontside", "carving"],
  "expanded_terms": ["stability", "stiffness", "damp", "edge_grip", "groomer", "carving"],
  "query_embedding_text": "stiff stable dampened ski for high-speed groomed run carving, not playful, at least 180cm"
}}

Example 3 - Feeling-based query:
Query: "Something surfy and buttery for deep days"
{{
  "desired_attributes": [
    {{"name": "powder_float", "weight": 0.95, "direction": "high"}},
    {{"name": "playfulness", "weight": 0.9, "direction": "high"}},
    {{"name": "stiffness", "weight": 0.7, "direction": "low"}}
  ],
  "negative_attributes": [],
  "spec_constraints": [],
  "categories": ["freeride", "powder"],
  "expanded_terms": ["powder_float", "playfulness", "surf", "soft", "deep snow"],
  "query_embedding_text": "soft playful ski with excellent powder flotation for deep snow days, surfy buttery feel"
}}

Now parse this query. Respond with JSON only:
{{
  "desired_attributes": [
    {{"name": "stiffness", "weight": 0.9, "direction": "high"}}
  ],
  "negative_attributes": [
    {{"name": "playfulness", "weight": 0.7, "direction": "low"}}
  ],
  "spec_constraints": [
    {{"field": "waist_width_mm", "op": ">=", "value": 105}}
  ],
  "categories": ["freeride"],
  "expanded_terms": ["edge_grip", "hardpack"],
  "query_embedding_text": "a rephrased version of the query using concrete attribute terms for embedding search"
}}"""
