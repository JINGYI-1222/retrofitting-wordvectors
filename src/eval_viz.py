from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab

WORD_GROUPS = {
    "positive emotion": ["happy", "joyful", "glad", "cheerful"],
    "negative emotion": ["sad", "unhappy", "miserable"],
    "intelligence":     ["smart", "intelligent", "clever"],
    "animals":          ["dog", "cat", "animal"],
    "movement":         ["run", "walk", "jog", "sprint"],
    "size":             ["big", "large", "huge", "small"],
    "speed":            ["fast", "rapid", "quick", "slow"],
}

GROUP_COLORS = {
    "positive emotion": "#E63946",
    "negative emotion": "#457B9D",
    "intelligence":     "#2A9D8F",
    "animals":          "#E9C46A",
    "movement":         "#9B5DE5",
    "size":             "#F77F00",
    "speed":            "#06D6A0",
}

if __name__ == "__main__":
    print("=== Loading GloVe ===")
    embeddings = load_text_embeddings(
        PROJECT_ROOT / "models/glove.6B.300d.txt", max_words=50000
    )

    print("=== Building WN_syn graph ===")
    graph = build_wordnet_graph(set(embeddings), include_synonyms=True)
    graph = filter_graph_by_vocab(graph, set(embeddings))

    print("=== Running retrofitting ===")
    retrofitted, _ = retrofit_vectors(embeddings, graph, num_iters=10, alpha=1.0)

    all_words = []
    word_to_group = {}
    for group, words in WORD_GROUPS.items():
        for w in words:
            if w in embeddings:
                all_words.append(w)
                word_to_group[w] = group

    print(f"Visualizing {len(all_words)} words")

    orig_vecs  = np.array([embeddings[w]  for w in all_words])
    retro_vecs = np.array([retrofitted[w] for w in all_words])

    combined = np.vstack([orig_vecs, retro_vecs])
    pca = PCA(n_components=2)
    pca.fit(combined)

    orig_2d  = pca.transform(orig_vecs)
    retro_2d = pca.transform(retro_vecs)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("white")
    ax.grid(color="#EEEEEE", linewidth=0.8)

    for i, word in enumerate(all_words):
        group = word_to_group[word]
        color = GROUP_COLORS[group]
        x0, y0 = orig_2d[i]
        x1, y1 = retro_2d[i]

        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
        ax.scatter(x0, y0, color=color, s=70, zorder=5)

        # Offset label to avoid overlap
        dx = x1 - x0
        dy = y1 - y0
        offset_x = -0.12 if dx < 0 else 0.12
        offset_y = -0.12 if dy < 0 else 0.12
        ax.text(x0 + offset_x, y0 + offset_y, word,
                fontsize=9, color=color, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    legend_patches = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g)
        for g in WORD_GROUPS
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10,
              framealpha=0.9)

    ax.set_title("Word Vector Movement After Retrofitting (PCA 2D)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig("eval_viz.png", dpi=150)
    print("Plot saved to eval_viz.png")
    plt.show()