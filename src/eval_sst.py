from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab


def sentence_to_vector(sentence: str, vectors: dict) -> np.ndarray | None:
    """Average word vectors for all known words in sentence."""
    words = sentence.lower().split()
    vecs = [vectors[w] for w in words if w in vectors]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def evaluate_sst(original_vectors: dict, retrofitted_vectors: dict) -> dict:
    """Train logistic regression on SST-2, compare original vs retrofitted."""
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("Loading SST-2 dataset...")
    ds = load_dataset("stanfordnlp/sst2")

    def encode(split, vectors):
        X, y = [], []
        for example in ds[split]:
            if example["label"] == -1:
                continue
            vec = sentence_to_vector(example["sentence"], vectors)
            if vec is not None:
                X.append(vec)
                y.append(example["label"])
        return np.array(X), np.array(y)

    print("Encoding sentences...")
    X_train_orig, y_train = encode("train",      original_vectors)
    X_val_orig,   y_val   = encode("validation", original_vectors)
    X_train_retro, _      = encode("train",      retrofitted_vectors)
    X_val_retro,   _      = encode("validation", retrofitted_vectors)

    print(f"Train size: {len(y_train)}, Val size: {len(y_val)}\n")

    print("Training on original GloVe...")
    clf_orig = LogisticRegression(max_iter=1000)
    clf_orig.fit(X_train_orig, y_train)
    acc_orig = accuracy_score(y_val, clf_orig.predict(X_val_orig))
    print(f"  Original accuracy:     {acc_orig:.4f}")

    print("Training on retrofitted vectors...")
    clf_retro = LogisticRegression(max_iter=1000)
    clf_retro.fit(X_train_retro, y_train)
    acc_retro = accuracy_score(y_val, clf_retro.predict(X_val_retro))
    print(f"  Retrofitted accuracy:  {acc_retro:.4f}\n")

    return {"before": acc_orig, "after": acc_retro}


def plot_sst(result: dict, save_path: str = "eval_sst.png"):
    labels = ["Original GloVe", "Retrofitted"]
    values = [result["before"], result["after"]]
    colors = ["steelblue", "tomato"]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_title("SST-2 Sentiment Classification: Before vs After Retrofitting")
    ax.set_ylim(0.5, 0.9)
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

    print("=== SST-2 Sentiment Analysis ===\n")
    sst_result = evaluate_sst(embeddings, retrofitted)
    plot_sst(sst_result)