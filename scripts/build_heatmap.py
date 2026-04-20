"""Build the README thumbnail: 12 designs x 5 query difficulties NDCG@5 heatmap."""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"
OUT = ROOT / "docs" / "thumbnail.png"

DIFFICULTIES = ["easy", "medium", "hard", "vague", "cross_domain"]
DIFF_LABELS = ["Easy", "Medium", "Hard", "Vague", "Cross-domain"]

PRETTY = {
    "design_01_graph": "01 · Knowledge Graph",
    "design_02_embedding": "02 · Pure Embedding",
    "design_03_llm_judge": "03 · LLM-as-Judge",
    "design_04_hybrid": "04 · Hybrid (SQL + Vec)",
    "design_05_sql": "05 · SQL + FTS5",
    "design_06_agents": "06 · Multi-Agent Pipeline",
    "design_07_bayesian": "07 · Bayesian",
    "design_08_tfidf": "08 · TF-IDF",
    "design_09_faceted": "09 · Faceted Search",
    "design_10_ensemble": "10 · Ensemble / LTR",
    "design_11_finetuned_embed": "11 · Fine-tuned Embeddings",
    "design_12_distilled_llm": "12 · Distilled LLM",
}


def collect() -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(RESULTS.glob("per_query/*.json")):
        d = json.loads(f.read_text())
        data[d["recommender"]][d["difficulty"]].append(d["metrics"]["ndcg_5"])
    summaries = list(RESULTS.glob("design_*/summary.json")) + [RESULTS / "summary.json"]
    for sumf in summaries:
        if not sumf.exists():
            continue
        for design in json.loads(sumf.read_text()):
            for q in design["per_query"]:
                if q["query_id"] in {qq["query_id"] for qq in design["per_query"]}:
                    data[design["name"]][q["difficulty"]].append(q["ndcg_5"])
    deduped: dict[str, dict[str, list[float]]] = {}
    for design, by_diff in data.items():
        deduped[design] = {diff: vals for diff, vals in by_diff.items()}
    return deduped


def main() -> None:
    data = collect()
    designs = sorted(data.keys())
    matrix = np.zeros((len(designs), len(DIFFICULTIES)))
    overall = []
    for i, design in enumerate(designs):
        row_vals = []
        for j, diff in enumerate(DIFFICULTIES):
            vals = data[design].get(diff, [])
            mean = float(np.mean(vals)) if vals else np.nan
            matrix[i, j] = mean
            if vals:
                row_vals.extend(vals)
        overall.append(float(np.mean(row_vals)) if row_vals else 0.0)

    order = sorted(range(len(designs)), key=lambda i: overall[i], reverse=True)
    designs = [designs[i] for i in order]
    matrix = matrix[order]
    overall = [overall[i] for i in order]

    labels = [PRETTY.get(d, d) for d in designs]

    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=160)
    cmap = plt.get_cmap("YlGnBu")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.85)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.2f}"
            color = "white" if v > 0.45 else "#222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(len(DIFF_LABELS)))
    ax.set_xticklabels(DIFF_LABELS, fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Query difficulty", fontsize=12, labelpad=10)
    ax.set_title(
        "Recommend-Arena · NDCG@5 by design × query difficulty\n"
        "12 recommendation systems, same dataset, same queries",
        fontsize=13,
        pad=14,
    )

    secax = ax.secondary_yaxis("right")
    secax.set_yticks(range(len(labels)))
    secax.set_yticklabels([f"{o:.3f}" for o in overall], fontsize=10)
    secax.set_ylabel("Overall NDCG@5", fontsize=11, labelpad=8)

    ax.set_xticks(np.arange(-0.5, len(DIFF_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.10)
    cbar.set_label("NDCG@5", fontsize=10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
