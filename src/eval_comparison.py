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


def load_ws353(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    print(f"Loaded {len(pairs)} pairs from WS-353")
    return pairs


def load_simlex(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[3])))
    print(f"Loaded {len(pairs)} pairs from SimLex-999")
    return pairs


def load_rg65(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    print(f"Loaded {len(pairs)} pairs from RG-65")
    return pairs


def cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def evaluate_word_similarity(vectors, pairs):
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


def plot_comparison(results, save_path="eval_wn_comparison.png"):
    """Bar chart: Original vs WN_syn vs WN_all for each dataset."""
    datasets = list(results.keys())
    original = [results[d]["original"] for d in datasets]
    wn_syn   = [results[d]["wn_syn"]   for d in datasets]
    wn_all   = [results[d]["wn_all"]   for d in datasets]

    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width, original, width, label="Original GloVe", color="#888888")
    bars2 = ax.bar(x,         wn_syn,   width, label="WN_syn",         color="#444444")
    bars3 = ax.bar(x + width, wn_all,   width, label="WN_all",         color="#111111")

    ax.set_ylabel("Spearman rho")
    ax.set_title("Word Similarity: Original vs WN_syn vs WN_all")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0.0, 1.0)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

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

    print("=== Building WN_syn graph (synonyms only) ===")
    graph_syn = build_wordnet_graph(set(embeddings), include_synonyms=True,
                                    include_hypernyms=False, include_hyponyms=False)
    graph_syn = filter_graph_by_vocab(graph_syn, set(embeddings))
    print(f"Graph nodes: {len(graph_syn)}\n")

    print("=== Running retrofitting WN_syn ===")
    retrofitted_syn, _ = retrofit_vectors(embeddings, graph_syn, num_iters=10, alpha=1.0)
    print("Done\n")

    print("=== Building WN_all graph (syn + hypernyms + hyponyms) ===")
    graph_all = build_wordnet_graph(set(embeddings), include_synonyms=True,
                                    include_hypernyms=True, include_hyponyms=True)
    graph_all = filter_graph_by_vocab(graph_all, set(embeddings))
    print(f"Graph nodes: {len(graph_all)}\n")

    print("=== Running retrofitting WN_all ===")
    retrofitted_all, _ = retrofit_vectors(embeddings, graph_all, num_iters=10, alpha=1.0)
    print("Done\n")

    ws353  = load_ws353(PROJECT_ROOT  / "datasets/combined.csv")
    simlex = load_simlex(PROJECT_ROOT / "datasets/SimLex-999.txt")
    rg65   = load_rg65(PROJECT_ROOT   / "datasets/rg65.txt")
    print()

    results = {}
    for name, pairs in [("WS-353", ws353), ("SimLex-999", simlex), ("RG-65", rg65)]:
        print(f"[{name}]")
        rho_orig, _ = evaluate_word_similarity(embeddings,      pairs)
        rho_syn,  _ = evaluate_word_similarity(retrofitted_syn, pairs)
        rho_all,  _ = evaluate_word_similarity(retrofitted_all, pairs)
        print(f"  Original: rho = {rho_orig:.4f}")
        print(f"  WN_syn:   rho = {rho_syn:.4f}")
        print(f"  WN_all:   rho = {rho_all:.4f}\n")
        results[name] = {"original": rho_orig, "wn_syn": rho_syn, "wn_all": rho_all}

    plot_comparison(results)