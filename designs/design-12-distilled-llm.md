# Design #12: Distilled LLM Ranker

## 1. Architecture Overview

This design captures a large LLM's recommendation reasoning in a model small enough to run on a laptop. The core mechanism is knowledge distillation: a teacher model (Claude or GPT-4o via the shared LLM provider) evaluates every (query, product) pair and produces structured judgments. A LoRA-tuned Qwen 2.5 student learns to reproduce those judgments. At query time, the student runs locally -- same structured output format, no API calls, no rate limits, no per-query cost.

The bet: a 0.5-1.5B parameter model, fine-tuned on domain-specific recommendation judgments, can match or approach a frontier LLM's ranking quality for the narrow task of "score this product against this query." We are not trying to build a general-purpose reasoner. We are compressing a specific capability -- product-preference matching with explanations -- into the smallest possible model.

This is a direct improvement on Design #3 (LLM-Judge). Design #3 calls a full LLM at query time for every candidate product, paying API cost and latency on every request. Design #12 front-loads all that reasoning into a one-time training phase, then serves from a tiny local model that runs in milliseconds per product.

### Training-Time Flow

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────────┐
│   Products   │────>│    Context    │────>│  (query, context)    │
│   + Reviews  │     │  Constructor  │     │   pair generator     │
└──────────────┘     └───────────────┘     └──────────┬───────────┘
                                                      │
                                                      v
                                           ┌──────────────────────┐
                                           │   Teacher LLM        │
                                           │   (Claude / GPT-4o)  │
                                           │                      │
                                           │   Input: query +     │
                                           │          product ctx  │
                                           │   Output: {score,    │
                                           │    explanation,       │
                                           │    matched_attrs}     │
                                           └──────────┬───────────┘
                                                      │
                                                      v
                                           ┌──────────────────────┐
                                           │  Instruction-tuning  │
                                           │  dataset (JSONL)     │
                                           │                      │
                                           │  input: prompt       │
                                           │  output: teacher JSON│
                                           └──────────┬───────────┘
                                                      │
                                                      v
                                           ┌──────────────────────┐
                                           │  LoRA fine-tune      │
                                           │  Qwen 2.5 (0.5B/    │
                                           │  1.5B) via unsloth   │
                                           └──────────┬───────────┘
                                                      │
                                                      v
                                           ┌──────────────────────┐
                                           │  LoRA adapter        │
                                           │  (~10-50MB)          │
                                           │  + optional GGUF     │
                                           └──────────────────────┘
```

### Query-Time Flow

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────────┐
│  User Query  │────>│  Candidate    │────>│  Student Model       │
│  (NL text)   │     │  Retrieval    │     │  (Qwen 2.5 + LoRA)  │
└──────────────┘     │  (all prods   │     │                      │
                     │  in domain)   │     │  For each product:   │
                     └───────────────┘     │  (query, context) -> │
                                           │  {score, explanation,│
                                           │   matched_attrs}     │
                                           └──────────┬───────────┘
                                                      │
                                                      v
                                           ┌──────────────────────┐
                                           │  Rank by score       │
                                           │  Return top-K with   │
                                           │  explanations        │
                                           └──────────────────────┘
```

### Why This Might Work

Language model distillation has been shown to work well when the student task is narrow. We are not asking the student to be a general conversationalist -- we are asking it to do one thing: given a product description and a user query, output a score and explain why. This is closer to a classification task with structured output than open-ended generation. Small models fine-tuned on narrow tasks regularly match larger models on those specific tasks.

### Why It Might Not

The teacher's quality is the ceiling. If Claude scores a product incorrectly during training, the student faithfully reproduces that error. There is no mechanism for the student to be smarter than the teacher. Additionally, distillation may not transfer well across domains -- a model trained on ski judgments might need retraining for running shoes, even if the teacher would handle both seamlessly.


## 2. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Teacher model** | Claude / GPT-4o via `shared.llm_provider` | Frontier-quality judgments; provider abstraction lets us swap models without code changes |
| **Student base model** | Qwen 2.5 (0.5B or 1.5B) | Small enough for local inference, strong instruction-following for its size, good structured output |
| **Fine-tuning framework** | `unsloth` | 2-5x faster LoRA training than stock HF, native Qwen 2.5 support, memory-efficient |
| **Adapter method** | LoRA (rank 16-64) via `peft` | Trains <1% of parameters, adapter is 10-50MB on disk, base model stays frozen |
| **Training data format** | JSONL (instruction-tuning) | Standard format for SFT, compatible with unsloth and HF trainers |
| **Product context store** | SQLite | Stores pre-built text summaries from reviews + specs; consistent with other designs |
| **Inference runtime** | `transformers` + `peft` (Python) | Direct integration, no external server needed |
| **Quantized inference** | GGUF export via `llama.cpp` / Ollama (optional) | 4-bit quantization cuts memory from 1.5GB to ~400MB for the 1.5B model; Ollama integration for serving |
| **Orchestration** | Plain Python | No framework overhead; direct model loading and inference |

### Why Qwen 2.5

The 0.5B and 1.5B variants hit a sweet spot for this use case. They are small enough to run on CPU in reasonable time (~200ms per inference for 0.5B) and support structured JSON output reliably after fine-tuning. Qwen 2.5's tokenizer handles English well, and the model family has strong unsloth support. The 0.5B variant is the first choice for POC -- if quality is insufficient, the 1.5B variant is a drop-in upgrade.

### Why Not a Smaller Approach

An obvious alternative is to skip the LLM entirely and train a cross-encoder or regression model on teacher labels. The reason to keep a generative model: we want natural language explanations and structured attribute breakdowns, not just scores. A regression model outputs a number; the distilled LLM outputs `{"score": 0.85, "explanation": "Strong match on stiffness and edge hold...", "matched_attributes": {"stiffness": 0.9, "edge_hold": 0.8}}`. The explanations are a core deliverable of the recommendation system, not a nice-to-have.


## 3. Data Model

### 3.1 Product Context

Each product is represented as a text summary built from its reviews and structured specs. This is the "document" the student model reads when evaluating a query-product pair.

```python
@dataclass
class ProductContext:
    """Pre-built text representation of a product for model input."""
    product_id: str
    product_name: str
    domain: str
    context_text: str        # concatenation of spec summary + review summary
    spec_summary: str        # "Length: 177cm, Waist: 100mm, Weight: 1850g, ..."
    review_summary: str      # LLM-generated consensus from reviews
    review_count: int
    metadata: dict           # raw specs, price, brand, year
```

### 3.2 Teacher Judgment

The structured output the teacher LLM produces for each (query, product) pair. This becomes the training label.

```python
@dataclass
class TeacherJudgment:
    """A single teacher evaluation of a (query, product) pair."""
    query: str
    product_id: str
    product_name: str
    score: float                          # 0.0 to 1.0
    explanation: str                      # 2-4 sentences explaining the match
    matched_attributes: dict[str, float]  # attribute -> match strength (0-1)
    teacher_model: str                    # "claude-sonnet-4-20250514", "gpt-4o", etc.
    timestamp: str                        # ISO 8601
```

### 3.3 Training Example

The instruction-tuning format fed to unsloth. Each example is a (input, output) pair where the output is the teacher's JSON response.

```python
@dataclass
class TrainingExample:
    """A single instruction-tuning example for the student model."""
    instruction: str    # system-level instruction (constant across examples)
    input: str          # "Rate this product for the query: {query}\n\nProduct: {context}"
    output: str         # teacher's JSON response as a string

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }
```

### 3.4 Student Inference Output

The structured output the student produces at query time. Identical schema to the teacher's output -- that is the entire point of distillation.

```python
@dataclass
class StudentJudgment:
    """Student model's evaluation of a (query, product) pair."""
    product_id: str
    product_name: str
    score: float                          # 0.0 to 1.0
    explanation: str
    matched_attributes: dict[str, float]
    raw_output: str                       # the model's raw text, for debugging
    parse_success: bool                   # whether JSON parsing succeeded
```

### 3.5 SQLite Schema

```sql
-- Pre-built product contexts
CREATE TABLE product_contexts (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    context_text TEXT NOT NULL,
    spec_summary TEXT,
    review_summary TEXT,
    review_count INTEGER DEFAULT 0,
    metadata_json TEXT,
    built_at TEXT NOT NULL
);

-- Teacher judgments (training labels)
CREATE TABLE teacher_judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    product_id TEXT NOT NULL REFERENCES product_contexts(product_id),
    score REAL NOT NULL,
    explanation TEXT NOT NULL,
    matched_attributes_json TEXT NOT NULL,
    teacher_model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(query, product_id, teacher_model)
);

-- Training run metadata
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model TEXT NOT NULL,          -- "Qwen/Qwen2.5-0.5B-Instruct"
    lora_rank INTEGER NOT NULL,
    num_examples INTEGER NOT NULL,
    num_epochs INTEGER NOT NULL,
    final_loss REAL,
    adapter_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_judgments_query ON teacher_judgments(query);
CREATE INDEX idx_judgments_product ON teacher_judgments(product_id);
CREATE INDEX idx_contexts_domain ON product_contexts(domain);
```


## 4. Teacher Labeling Pipeline

The teacher pipeline generates high-quality training labels by running a frontier LLM on every (query, product) pair. This is the most expensive step in the system and runs once (or infrequently, when new products or queries are added).

### 4.1 Prompt Template

The teacher prompt is designed to produce deterministic, structured output. It includes the scoring rubric, desired output format, and domain context.

```python
TEACHER_PROMPT = """You are an expert {domain} product recommender. A user has described what they want, and you need to evaluate how well a specific product matches their preferences.

User's query: "{query}"

Product: {product_name}
{product_context}

Evaluate this product against the user's query. Consider:
- How well each mentioned attribute matches the user's stated preferences
- Trade-offs: where the product excels vs. falls short
- Overall suitability, accounting for how important each attribute is to the query

Respond with ONLY valid JSON in this exact format:
{{
  "score": <float 0.0 to 1.0, where 1.0 is a perfect match>,
  "explanation": "<2-4 sentences explaining why this score, citing specific product characteristics>",
  "matched_attributes": {{
    "<attribute_name>": <float 0.0 to 1.0 indicating match strength>,
    ...
  }}
}}

Scoring guidelines:
- 0.9-1.0: Near-perfect match on all queried attributes
- 0.7-0.89: Strong match with minor gaps
- 0.5-0.69: Decent match but notable trade-offs
- 0.3-0.49: Partial match, significant misalignments
- 0.0-0.29: Poor match, wrong category or contradicts preferences"""
```

### 4.2 Labeling Logic

```python
def label_pair(
    query: str,
    product_ctx: ProductContext,
    llm: LLMProvider,
    domain: str,
) -> TeacherJudgment:
    """Score a single (query, product) pair using the teacher model."""
    prompt = TEACHER_PROMPT.format(
        domain=domain,
        query=query,
        product_name=product_ctx.product_name,
        product_context=product_ctx.context_text,
    )
    response = llm.generate(prompt, json_mode=True)
    parsed = json.loads(response)

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, float(parsed["score"])))

    return TeacherJudgment(
        query=query,
        product_id=product_ctx.product_id,
        product_name=product_ctx.product_name,
        score=score,
        explanation=parsed["explanation"],
        matched_attributes={
            k: max(0.0, min(1.0, float(v)))
            for k, v in parsed.get("matched_attributes", {}).items()
        },
        teacher_model=llm.llm_model,
        timestamp=datetime.utcnow().isoformat(),
    )
```

### 4.3 Batch Labeling

For the benchmark dataset (25 products, 20 test queries), the teacher evaluates all 500 (query, product) pairs. With rate limiting and API costs in mind, the batch labeler processes pairs sequentially with caching to support resumption.

```python
def label_all_pairs(
    queries: list[str],
    products: list[ProductContext],
    llm: LLMProvider,
    domain: str,
    db: sqlite3.Connection,
) -> list[TeacherJudgment]:
    """Label all (query, product) pairs, skipping cached results."""
    judgments = []
    total = len(queries) * len(products)
    completed = 0

    for query in queries:
        for product in products:
            # Check cache
            cached = db.execute(
                "SELECT score, explanation, matched_attributes_json "
                "FROM teacher_judgments WHERE query=? AND product_id=?",
                (query, product.product_id),
            ).fetchone()

            if cached:
                completed += 1
                continue

            judgment = label_pair(query, product, llm, domain)

            # Store to SQLite
            db.execute(
                "INSERT INTO teacher_judgments "
                "(query, product_id, score, explanation, "
                "matched_attributes_json, teacher_model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    query, product.product_id, judgment.score,
                    judgment.explanation,
                    json.dumps(judgment.matched_attributes),
                    judgment.teacher_model, judgment.timestamp,
                ),
            )
            db.commit()
            judgments.append(judgment)

            completed += 1
            if completed % 50 == 0:
                logger.info("Labeled %d/%d pairs", completed, total)

    return judgments
```

### 4.4 Cost Estimate

For 25 products x 20 queries = 500 pairs:
- **Claude (Sonnet):** ~500 tokens input + ~200 tokens output per pair = ~350K total tokens. At current pricing, roughly $0.50-1.00 total.
- **GPT-4o:** Similar token count, similar cost.
- **Time:** ~2-5 minutes with rate limiting.

This is a one-time cost. Relabeling is only needed when products or queries change.

### 4.5 Synthetic Data Augmentation

For benchmark evaluation, 500 examples may be thin for fine-tuning. Synthetic augmentation generates additional training pairs by paraphrasing queries and interpolating product contexts. This is used only when the real data is insufficient -- the teacher labels on real data are always the primary training signal.

```python
AUGMENTATION_PROMPTS = {
    "query_paraphrase": (
        "Rephrase this product query in a different way, "
        "keeping the same intent:\n\n{query}\n\n"
        "Return only the rephrased query."
    ),
    "attribute_focus": (
        "Write a product query that specifically asks for a {domain} "
        "product with these characteristics: {attributes}\n\n"
        "Return only the query."
    ),
}

def augment_queries(
    original_queries: list[str],
    llm: LLMProvider,
    domain: str,
    multiplier: int = 3,
) -> list[str]:
    """Generate synthetic query variants for training augmentation."""
    augmented = list(original_queries)  # keep originals
    for query in original_queries:
        for _ in range(multiplier):
            prompt = AUGMENTATION_PROMPTS["query_paraphrase"].format(query=query)
            paraphrase = llm.generate(prompt).strip()
            augmented.append(paraphrase)
    return augmented
```


## 5. Student Training Pipeline

### 5.1 Dataset Preparation

Teacher judgments are converted to instruction-tuning format. Each example teaches the student to produce the same structured JSON the teacher produced, given the same input.

```python
STUDENT_INSTRUCTION = (
    "You are a product recommendation assistant. Given a user query and a "
    "product description, evaluate how well the product matches the query. "
    "Respond with valid JSON containing score, explanation, and "
    "matched_attributes."
)

def build_training_dataset(
    judgments: list[TeacherJudgment],
    contexts: dict[str, ProductContext],
) -> list[dict]:
    """Convert teacher judgments to instruction-tuning format."""
    dataset = []
    for j in judgments:
        ctx = contexts[j.product_id]
        input_text = (
            f"Rate this product for the query: {j.query}\n\n"
            f"Product: {ctx.product_name}\n"
            f"{ctx.context_text}"
        )
        output_text = json.dumps({
            "score": round(j.score, 2),
            "explanation": j.explanation,
            "matched_attributes": {
                k: round(v, 2) for k, v in j.matched_attributes.items()
            },
        }, indent=2)

        dataset.append({
            "instruction": STUDENT_INSTRUCTION,
            "input": input_text,
            "output": output_text,
        })
    return dataset
```

### 5.2 LoRA Configuration

The LoRA config targets the attention and MLP projection layers. Rank 16 is the default -- enough to capture the narrow task without overfitting on 500-2000 examples.

```python
from peft import LoraConfig

LORA_CONFIG = LoraConfig(
    r=16,                        # rank: 16 for POC, 32-64 if quality is lacking
    lora_alpha=32,               # scaling factor, typically 2x rank
    target_modules=[             # Qwen 2.5 attention + MLP projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,           # light regularization
    bias="none",
    task_type="CAUSAL_LM",
)
```

### 5.3 Training Script

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import json

def train_student(
    dataset_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "adapters/design-12",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """Fine-tune Qwen 2.5 on teacher judgments via LoRA."""
    # Load base model with unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,              # auto-detect (float16 on GPU, float32 on CPU)
        load_in_4bit=True,       # QLoRA: 4-bit base + float16 LoRA
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Format as chat/instruction template
    def format_example(example):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
        return {"text": prompt}

    dataset = dataset.map(format_example)

    # Training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=10,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            optim="adamw_8bit",
        ),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
```

### 5.4 Training Regime

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen 2.5 0.5B Instruct | Smallest viable; upgrade to 1.5B if quality insufficient |
| LoRA rank | 16 | Sufficient for narrow task; rank 32-64 for harder domains |
| Epochs | 3 | Small dataset -- more epochs risk overfitting, fewer risk underfitting |
| Batch size | 4 (effective 16 with gradient accumulation) | Fits in 8GB VRAM |
| Learning rate | 2e-4 | Standard for LoRA SFT |
| Max sequence length | 2048 | Sufficient for query + product context + JSON output |
| Quantization | QLoRA (4-bit base) | Cuts training memory from ~4GB to ~2GB for 0.5B |
| Training time | ~5-15 minutes on a single GPU | For 500-2000 examples; Apple Silicon M-series or any CUDA GPU |

### 5.5 Export Options

After training, the adapter can be used in three modes:

**1. Python (transformers + peft):** Load base model + adapter directly. Best for development and benchmarking.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "adapters/design-12")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

**2. Merged model:** Merge LoRA weights into the base model for simpler deployment.

```python
merged = model.merge_and_unload()
merged.save_pretrained("models/design-12-merged")
```

**3. GGUF (llama.cpp / Ollama):** Quantize the merged model to GGUF format for maximum portability and minimal memory.

```bash
# Convert merged model to GGUF (requires llama.cpp)
python convert_hf_to_gguf.py models/design-12-merged --outtype q4_k_m \
    --outfile models/design-12.Q4_K_M.gguf

# Create Ollama model
cat > Modelfile <<EOF
FROM models/design-12.Q4_K_M.gguf
SYSTEM "You are a product recommendation assistant..."
EOF
ollama create design-12 -f Modelfile
```


## 6. Ingestion Pipeline

Ingestion builds the product context store -- text summaries that the student model reads at query time. This is separate from the training pipeline: ingestion happens first, training uses the ingested contexts.

### 6.1 Context Construction

For each product, the ingestion pipeline builds a text summary from structured specs and reviews.

```
Raw Products + Reviews
    |
    v
[1] Extract specs from product metadata
    |        "Length: 177cm, Waist: 100mm, Weight: 1850g"
    |
    v
[2] Summarize reviews via LLM (or concatenate top-K)
    |        "Reviewers consistently praise edge hold on hardpack..."
    |
    v
[3] Assemble context text
    |        specs + review summary + notable quotes
    |
    v
[4] Store in SQLite (product_contexts table)
```

```python
def build_product_context(
    product: dict,
    reviews: list[dict],
    llm: LLMProvider,
    domain: str,
) -> ProductContext:
    """Build a text context for a product from its specs and reviews."""
    # Spec summary from metadata
    spec_parts = []
    for key, value in product.get("metadata", {}).items():
        spec_parts.append(f"{key.replace('_', ' ').title()}: {value}")
    spec_summary = ", ".join(spec_parts) if spec_parts else "No specs available."

    # Review summary via LLM
    review_texts = [r["review_text"] for r in reviews]
    if review_texts:
        review_block = "\n---\n".join(review_texts[:20])  # cap at 20 reviews
        summary_prompt = (
            f"Summarize what reviewers say about this {domain} product: "
            f'"{product["product_name"]}".\n\n'
            f"Reviews:\n{review_block}\n\n"
            f"Write a 3-5 sentence summary covering the key attributes "
            f"reviewers mention (performance, feel, strengths, weaknesses). "
            f"Be specific -- use the reviewers' language."
        )
        review_summary = llm.generate(summary_prompt)
    else:
        review_summary = "No reviews available."

    # Assemble full context
    context_text = (
        f"Specs: {spec_summary}\n\n"
        f"Review consensus ({len(review_texts)} reviews): {review_summary}"
    )

    return ProductContext(
        product_id=product["product_id"],
        product_name=product["product_name"],
        domain=domain,
        context_text=context_text,
        spec_summary=spec_summary,
        review_summary=review_summary,
        review_count=len(review_texts),
        metadata=product.get("metadata", {}),
    )
```

### 6.2 Recommender.ingest() Implementation

The `ingest` method builds contexts for all products and stores them. If a teacher model is available, it can also trigger the labeling pipeline -- but labeling is optional at ingest time and can be run separately.

```python
def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
    """Build product contexts from reviews + specs and store them."""
    reviews_by_product = defaultdict(list)
    for r in reviews:
        reviews_by_product[r["product_id"]].append(r)

    for product in products:
        pid = product["product_id"]
        product_reviews = reviews_by_product.get(pid, [])

        ctx = build_product_context(
            product, product_reviews, self.llm, domain
        )
        self.contexts[pid] = ctx
        self.product_names[pid] = product["product_name"]

        # Store in SQLite
        self.db.execute(
            "INSERT OR REPLACE INTO product_contexts "
            "(product_id, product_name, domain, context_text, "
            "spec_summary, review_summary, review_count, metadata_json, built_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid, product["product_name"], domain,
                ctx.context_text, ctx.spec_summary, ctx.review_summary,
                ctx.review_count, json.dumps(ctx.metadata),
                datetime.utcnow().isoformat(),
            ),
        )
    self.db.commit()
```

### 6.3 Throughput

For 25 products with an average of 5 reviews each:
- **Context construction with LLM summarization:** ~1-2 seconds per product via Ollama, ~0.5s via API. Total: ~30-60 seconds.
- **Storage:** Negligible. Each context is 500-2000 characters.

Ingestion is fast and cheap. The expensive step is teacher labeling (Section 4), which is separate.


## 7. Query / Ranking Pipeline

At query time, the student model evaluates each candidate product and produces a structured judgment. No API calls, no network latency -- the model runs in-process.

### 7.1 Inference Logic

```python
def query(
    self,
    query_text: str,
    domain: str,
    top_k: int = 10,
) -> list[RecommendationResult]:
    """Score all products in a domain using the distilled student model."""
    # Get all product contexts for this domain
    candidates = [
        ctx for ctx in self.contexts.values()
        if ctx.domain == domain or domain == ""
    ]

    # Score each candidate
    scored = []
    for ctx in candidates:
        judgment = self._infer_student(query_text, ctx)
        scored.append(judgment)

    # Sort by score descending
    scored.sort(key=lambda j: j.score, reverse=True)

    # Convert to RecommendationResult
    results = []
    for j in scored[:top_k]:
        results.append(RecommendationResult(
            product_id=j.product_id,
            product_name=j.product_name,
            score=round(j.score, 4),
            explanation=j.explanation,
            matched_attributes=j.matched_attributes,
        ))
    return results
```

### 7.2 Student Model Inference

```python
def _infer_student(
    self,
    query: str,
    product_ctx: ProductContext,
) -> StudentJudgment:
    """Run the student model on a single (query, product) pair."""
    prompt = (
        f"### Instruction:\n{STUDENT_INSTRUCTION}\n\n"
        f"### Input:\n"
        f"Rate this product for the query: {query}\n\n"
        f"Product: {product_ctx.product_name}\n"
        f"{product_ctx.context_text}\n\n"
        f"### Response:\n"
    )

    inputs = self.tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=self.max_seq_length,
    ).to(self.model.device)

    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,         # deterministic
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)

    # Parse JSON from response
    try:
        parsed = json.loads(self._extract_json(raw_output))
        return StudentJudgment(
            product_id=product_ctx.product_id,
            product_name=product_ctx.product_name,
            score=max(0.0, min(1.0, float(parsed["score"]))),
            explanation=parsed.get("explanation", ""),
            matched_attributes={
                k: max(0.0, min(1.0, float(v)))
                for k, v in parsed.get("matched_attributes", {}).items()
            },
            raw_output=raw_output,
            parse_success=True,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(
            "Failed to parse student output for %s: %s",
            product_ctx.product_name, e,
        )
        return StudentJudgment(
            product_id=product_ctx.product_id,
            product_name=product_ctx.product_name,
            score=0.0,
            explanation=f"Parse error: {e}",
            matched_attributes={},
            raw_output=raw_output,
            parse_success=False,
        )

def _extract_json(self, text: str) -> str:
    """Extract the first JSON object from model output."""
    import re
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # Find first { ... } block
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
                return text[start:i + 1]
    raise json.JSONDecodeError("Unterminated JSON object", text, start)
```

### 7.3 Batched Inference

For the benchmark's 25 products, sequential inference on the 0.5B model takes ~5-10 seconds total on Apple Silicon (200-400ms per product). This is already fast enough for the POC. For larger catalogs, batched inference with padding is straightforward:

```python
def _infer_batch(
    self,
    query: str,
    product_ctxs: list[ProductContext],
    batch_size: int = 8,
) -> list[StudentJudgment]:
    """Batch inference for multiple products against the same query."""
    all_judgments = []
    for i in range(0, len(product_ctxs), batch_size):
        batch = product_ctxs[i:i + batch_size]
        prompts = [self._build_prompt(query, ctx) for ctx in batch]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_seq_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        for j, ctx in enumerate(batch):
            generated = outputs[j][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            all_judgments.append(self._parse_output(raw, ctx))

    return all_judgments
```

### 7.4 Latency Profile

| Stage | Time (0.5B, Apple M2) | Time (1.5B, Apple M2) |
|-------|----------------------|----------------------|
| Tokenize prompt | ~1ms | ~1ms |
| Generate (per product) | ~200ms | ~600ms |
| Parse JSON | <1ms | <1ms |
| Total for 25 products | ~5s | ~15s |
| Total for 25 products (batched) | ~3s | ~10s |

Compare to Design #3 (LLM-Judge): 10-25 seconds for 20 products with Ollama, and that requires a running Ollama server. Design #12 runs in-process with no external dependencies beyond the model files on disk.


## 8. Benchmark Integration

### 8.1 Recommender Protocol Implementation

```python
from shared.interface import Recommender, RecommendationResult
from shared.llm_provider import get_provider

import json
import logging
import sqlite3
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class DistilledLLMRecommender:
    """Implements the Recommender protocol using a distilled student model.

    Training-time: uses a teacher LLM to label (query, product) pairs.
    Query-time: uses a LoRA-tuned Qwen 2.5 student model locally.
    """

    def __init__(
        self,
        adapter_path: str = "adapters/design-12",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        db_path: str = "design_12.db",
        device: str = "auto",
    ):
        self.adapter_path = Path(adapter_path)
        self.base_model_name = base_model
        self.db = sqlite3.connect(db_path)
        self._init_db()

        self.contexts: dict[str, ProductContext] = {}
        self.product_names: dict[str, str] = {}
        self.llm = get_provider()  # teacher, used during ingestion/labeling

        # Student model loaded lazily (only needed at query time)
        self._model = None
        self._tokenizer = None
        self.device = device
        self.max_seq_length = 2048

    def _init_db(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS product_contexts (
                product_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                domain TEXT NOT NULL,
                context_text TEXT NOT NULL,
                spec_summary TEXT,
                review_summary TEXT,
                review_count INTEGER DEFAULT 0,
                metadata_json TEXT,
                built_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS teacher_judgments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                product_id TEXT NOT NULL,
                score REAL NOT NULL,
                explanation TEXT NOT NULL,
                matched_attributes_json TEXT NOT NULL,
                teacher_model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(query, product_id, teacher_model)
            );
        """)

    @property
    def model(self):
        if self._model is None:
            self._load_student()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_student()
        return self._tokenizer

    def _load_student(self):
        """Load base model + LoRA adapter."""
        logger.info("Loading student model: %s + %s", self.base_model_name, self.adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        if self.adapter_path.exists():
            self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        else:
            logger.warning(
                "Adapter not found at %s; using base model (untrained). "
                "Run the training pipeline first.",
                self.adapter_path,
            )
            self._model = base
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        # Implementation as shown in Section 6.2
        ...

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        # Implementation as shown in Section 7.1
        ...
```

### 8.2 Benchmark Workflow

The benchmark runner interacts with Design #12 identically to every other design:

```python
from design_12 import DistilledLLMRecommender

rec = DistilledLLMRecommender(
    adapter_path="adapters/design-12-ski",
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
)

# Ingestion (builds product contexts)
rec.ingest(products=catalog, reviews=reviews, domain="ski")

# Query (uses student model, no API calls)
results = rec.query("stiff damp carving ski for hardpack", domain="ski", top_k=10)

for r in results:
    assert 0.0 <= r.score <= 1.0
    assert isinstance(r.matched_attributes, dict)
    assert len(r.explanation) > 0
```

### 8.3 Pre-Training Checklist

Before the benchmark can run, the training pipeline must be completed:

1. **Ingest products** to build product contexts (Section 6).
2. **Run teacher labeling** on all (query, product) pairs (Section 4).
3. **Build training dataset** from teacher judgments (Section 5.1).
4. **Fine-tune student** via unsloth (Section 5.3).
5. **Verify adapter loads** and produces valid JSON output.

Steps 1-4 are one-time costs. Step 5 is a sanity check. The benchmark itself only runs steps from the `ingest` and `query` methods -- the training pipeline is a separate prerequisite.

### 8.4 Fallback Behavior

If the student model produces unparseable output for a product (JSON parse failure), the system assigns a score of 0.0 with an explanation noting the parse error. This ensures the benchmark always gets a complete result set. The `parse_success` field on `StudentJudgment` tracks how often this happens -- a high failure rate indicates the student needs more training data or a higher-rank LoRA.


## 9. Trade-offs and Limitations

### Strengths

- **No per-query API cost.** After training, all inference is local. A benchmark run of 20 queries x 25 products = 500 inferences costs $0.
- **Fast inference.** 200-600ms per product on Apple Silicon, 3-15 seconds for a full ranking pass. Compare to Design #3's 10-25 seconds with LLM judge calls.
- **Structured explanations.** Unlike pure-numeric rankers (Designs #2, #8, #11), every recommendation comes with natural language reasoning and per-attribute match scores.
- **Portable.** The LoRA adapter is 10-50MB. The base model is 500MB-1.5GB. The entire system fits on a USB drive and runs offline.
- **Reproducible.** Deterministic inference (temperature=0) means the same query always produces the same ranking. No LLM-induced variance between benchmark runs.
- **Minimal infrastructure.** Python + SQLite + a model file. No vector database, no Ollama server, no API keys at query time.

### Weaknesses

- **Teacher quality ceiling.** The student can never exceed the teacher's judgment quality. If Claude consistently mis-scores a product, the student faithfully reproduces that error. There is no self-correction mechanism.
- **Training cost is front-loaded.** The teacher labeling step requires API calls (~$0.50-1.00 for the benchmark dataset). This is cheap per-run but adds a dependency on an API provider during the training phase.
- **Domain transfer is uncertain.** A model trained on ski judgments may not generalize to running shoes without retraining. The teacher would handle both domains seamlessly; the student may need per-domain adapters. This is an empirical question the POC must answer.
- **Cold start on new queries.** The student has only seen the query distribution from training. A radically different query style (e.g., "I want the opposite of my current ski") may produce lower-quality judgments than the teacher would. Mitigation: synthetic query augmentation during training.
- **Model size is non-trivial.** The 0.5B model requires ~1GB of disk and ~1-2GB of RAM at inference. For a benchmark on a laptop this is fine; for edge deployment it could be a constraint.
- **JSON parse failures.** Small models occasionally produce malformed JSON, especially for edge-case inputs. The fallback (score 0.0) is safe but degrades ranking quality. Mitigation: constrained decoding (e.g., `outlines` library) to force valid JSON.
- **No real-time learning.** The student is frozen after training. It cannot adapt to user feedback at query time. Design #7 (Bayesian) and Design #10 (Ensemble LTR) can incorporate feedback incrementally.

### Compared to Other Designs

| Dimension | Design #3 (LLM-Judge) | Design #12 (Distilled) | Winner |
|-----------|----------------------|----------------------|--------|
| Query latency | 10-25s (API calls) | 3-15s (local) | #12 |
| Per-query cost | $0.001-0.01 | $0 | #12 |
| Ranking quality | Teacher quality | <= Teacher quality | #3 (ceiling) |
| Explanations | Yes (full LLM) | Yes (distilled) | Tie |
| Setup complexity | Low (just API key) | High (training pipeline) | #3 |
| Offline capable | No (needs API) | Yes | #12 |
| Adapts to new domains | Instantly | Needs retraining | #3 |

| Dimension | Design #11 (Fine-Tuned Embed) | Design #12 (Distilled) | Winner |
|-----------|------------------------------|----------------------|--------|
| Inference speed | Sub-millisecond | 200-600ms/product | #11 |
| Explanations | Centroid decomposition | Full NL + attributes | #12 |
| Training data | Contrastive pairs | Teacher judgments | Different |
| Model size | ~50MB | ~500MB-1.5GB | #11 |

| Dimension | Design #6 (Multi-Agent) | Design #12 (Distilled) | Winner |
|-----------|------------------------|----------------------|--------|
| Query latency | 30-60s (multiple LLM) | 3-15s (local) | #12 |
| Reasoning depth | Multi-perspective | Single-pass | #6 |
| Cost per query | $0.05-0.10 | $0 | #12 |
| Complexity | High (orchestration) | Moderate (training) | #12 |


## 10. Future Directions

### 10.1 Multi-Turn Distillation

The current design handles single-turn queries. A natural extension: distill multi-turn recommendation dialogues where the teacher refines its judgment based on follow-up questions ("That's too expensive -- what about under $500?"). The student would learn to maintain context across turns and update its scoring.

### 10.2 Preference Learning from User Feedback

After deployment, user feedback (clicks, "this was helpful", explicit ratings) can be used to fine-tune the student further. This creates a feedback loop:

```
Teacher labels (initial) --> Student v1
                             + user feedback --> Student v2
                             + more feedback --> Student v3 ...
```

The teacher provides the initial bootstrap; user feedback refines domain-specific preferences that the teacher might not capture. This could be implemented as periodic LoRA re-training on a mix of teacher labels and user preference data (DPO or RLHF-style).

### 10.3 GGUF / Ollama Deployment

For production use, the merged + quantized GGUF model can be served via Ollama with a custom Modelfile. This gives:
- Drop-in replacement for any Ollama-based system
- Standard API for inference
- Easy model versioning and distribution
- Compatibility with llama.cpp ecosystem tools

### 10.4 Constrained Decoding

The current JSON parsing relies on the model producing well-formed JSON. A more robust approach: use constrained decoding (via `outlines`, `guidance`, or `llama.cpp` grammars) to guarantee the output conforms to the expected schema. This eliminates parse failures entirely at the cost of slightly slower inference.

```python
from outlines import models, generate

schema = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "explanation": {"type": "string"},
        "matched_attributes": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
    "required": ["score", "explanation", "matched_attributes"],
}

model = models.transformers("Qwen/Qwen2.5-0.5B-Instruct")
generator = generate.json(model, schema)
result = generator(prompt)  # guaranteed valid JSON matching schema
```

### 10.5 Teacher Ensemble

Instead of a single teacher, use multiple teacher models (Claude + GPT-4o + Gemini) and average their judgments. This reduces the risk of single-teacher bias and produces more robust training labels. The cost triples, but for a 500-pair dataset this is still under $5.

### 10.6 Per-Domain Adapters with Shared Base

If domain transfer proves insufficient, train separate LoRA adapters for each domain while sharing the same base model. At query time, load the appropriate adapter based on the `domain` parameter. This keeps the base model download to a one-time cost while allowing domain specialization.

```python
DOMAIN_ADAPTERS = {
    "ski": "adapters/design-12-ski",
    "running_shoe": "adapters/design-12-shoe",
    "cookie": "adapters/design-12-cookie",
}

def _load_adapter_for_domain(self, domain: str):
    adapter_path = DOMAIN_ADAPTERS.get(domain)
    if adapter_path and Path(adapter_path).exists():
        self._model = PeftModel.from_pretrained(self._base_model, adapter_path)
    else:
        logger.warning("No adapter for domain %s, using base model", domain)
```

### 10.7 Active Learning for Efficient Labeling

Instead of labeling all (query, product) pairs, use active learning to select the most informative pairs for the teacher to label. Products where the student is uncertain (low confidence or high variance across augmented queries) are prioritized for teacher labeling. This could reduce teacher API costs by 50-70% while maintaining quality.
