from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab


def load_ws353(path: str) -> list:
    """Load WS-353: word1,word2,score (with header)"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    print(f"Loaded {len(pairs)} pairs from WS-353")
    return pairs


def load_simlex(path: str) -> list:
    """Load SimLex-999: tab separated, column 4 is score (with header)"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[3])))
    print(f"Loaded {len(pairs)} pairs from SimLex-999")
    return pairs


def load_rg65(path: str) -> list:
    """Load RG-65: tab separated, no header"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    print(f"Loaded {len(pairs)} pairs from RG-65")
    return pairs


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def evaluate_word_similarity(vectors: dict, pairs: list) -> tuple:
    """Compute Spearman rho between model cosine similarity and human scores."""
    human_scores, model_scores = [], []
    skipped = 0
    for w1, w2, human in pairs:
        if w1 not in vectors or w2 not in vectors:
            skipped += 1
            continue
        model_scores.append(cosine_similarity(vectors[w1], vectors[w2]))
        human_scores.append(human)
    print(f"  Coverage: {len(human_scores)}/{len(pairs)}, skipped {skipped} pairs")
    if len(human_scores) < 2:
        return 0.0, []
    rho, _ = spearmanr(human_scores, model_scores)
    return rho, list(zip(human_scores, model_scores))


def plot_comparison(results: dict, save_path: str = "eval_comparison.png"):
    """Bar chart: before vs after retrofitting for each dataset."""
    datasets = list(results.keys())
    before = [results[d]["before"] for d in datasets]
    after  = [results[d]["after"]  for d in datasets]
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before, width, label="Original GloVe", color="steelblue")
    bars2 = ax.bar(x + width/2, after,  width, label="Retrofitted",    color="tomato")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Word Similarity: Before vs After Retrofitting")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0.0, 1.0)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    print("=== Loading GloVe (first 50000 words) ===")
    embeddings = load_text_embeddings(
        PROJECT_ROOT / "models/glove.6B.300d.txt", max_words=50000
    )
    print(f"Loaded {len(embeddings)} embeddings\n")

    print("=== Building WordNet graph ===")
    graph = build_wordnet_graph(set(embeddings), include_synonyms=True)
    graph = filter_graph_by_vocab(graph, set(embeddings))
    print(f"Graph nodes: {len(graph)}\n")

    print("=== Running retrofitting (10 iterations) ===")
    retrofitted, stats = retrofit_vectors(embeddings, graph, num_iters=10, alpha=1.0)
    print("Retrofit stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    # Load datasets
    ws353   = load_ws353(PROJECT_ROOT  / "datasets/combined.csv")
    simlex  = load_simlex(PROJECT_ROOT / "datasets/SimLex-999.txt")
    rg65    = load_rg65(PROJECT_ROOT   / "datasets/rg65.txt")
    print()

    # Evaluate
    results = {}
    for name, pairs in [("WS-353", ws353), ("SimLex-999", simlex), ("RG-65", rg65)]:
        print(f"[{name}]")
        rho_before, _ = evaluate_word_similarity(embeddings,   pairs)
        rho_after,  _ = evaluate_word_similarity(retrofitted,  pairs)
        print(f"  Before: rho = {rho_before:.4f}")
        print(f"  After:  rho = {rho_after:.4f}\n")
        results[name] = {"before": rho_before, "after": rho_after}

    plot_comparison(results)