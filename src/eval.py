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
    """Load WS-353 dataset as list of (word1, word2, score)."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            w1, w2, score = parts[0].lower(), parts[1].lower(), float(parts[2])
            pairs.append((w1, w2, score))
    print(f"Loaded {len(pairs)} pairs from WS-353")
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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, before, width, label="Original",    color="steelblue")
    ax.bar(x + width/2, after,  width, label="Retrofitted", color="tomato")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Word Similarity: Before vs After Retrofitting")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    print("=== Loading GloVe embeddings (first 50000 words) ===\n")
    embeddings = load_text_embeddings(
        PROJECT_ROOT / "models/glove.6B.300d.txt",
        max_words=50000
    )
    print(f"Loaded {len(embeddings)} embeddings\n")

    print("=== Building WordNet graph ===\n")
    graph = build_wordnet_graph(set(embeddings), include_synonyms=True)
    graph = filter_graph_by_vocab(graph, set(embeddings))
    print(f"Graph nodes: {len(graph)}\n")

    print("=== Running retrofitting (10 iterations) ===\n")
    retrofitted, stats = retrofit_vectors(
        embeddings, graph, num_iters=10, alpha=1.0
    )
    print("Retrofit stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    print("=== Evaluating on WS-353 ===\n")
    pairs = load_ws353(PROJECT_ROOT / "datasets/combined.csv")

    print("[Original GloVe]")
    rho_before, _ = evaluate_word_similarity(embeddings, pairs)
    print(f"  Spearman rho = {rho_before:.4f}\n")

    print("[Retrofitted]")
    rho_after, _ = evaluate_word_similarity(retrofitted, pairs)
    print(f"  Spearman rho = {rho_after:.4f}\n")

    plot_comparison({
        "WS-353": {"before": rho_before, "after": rho_after}
    })