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


def sentence_to_vector(sentence: str, vectors: dict):
    words = sentence.lower().split()
    vecs = [vectors[w] for w in words if w in vectors]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def evaluate_sst(original, syn, all_):
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("Loading SST-2...")
    ds = load_dataset("stanfordnlp/sst2")

    def encode(split, vectors):
        X, y = [], []
        for ex in ds[split]:
            if ex["label"] == -1:
                continue
            vec = sentence_to_vector(ex["sentence"], vectors)
            if vec is not None:
                X.append(vec)
                y.append(ex["label"])
        return np.array(X), np.array(y)

    print("Encoding sentences...")
    X_train_orig, y_train = encode("train",      original)
    X_val_orig,   y_val   = encode("validation", original)
    X_train_syn,  _       = encode("train",      syn)
    X_val_syn,    _       = encode("validation", syn)
    X_train_all,  _       = encode("train",      all_)
    X_val_all,    _       = encode("validation", all_)

    print(f"Train: {len(y_train)}, Val: {len(y_val)}\n")

    def train_eval(X_tr, X_val, label):
        print(f"Training {label}...")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"  {label} accuracy: {acc:.4f}")
        return acc

    acc_orig = train_eval(X_train_orig, X_val_orig, "Original GloVe")
    acc_syn  = train_eval(X_train_syn,  X_val_syn,  "WN_syn")
    acc_all  = train_eval(X_train_all,  X_val_all,  "WN_all")

    return {"original": acc_orig, "wn_syn": acc_syn, "wn_all": acc_all}


def plot_sst(result, save_path="eval_sst_comparison.png"):
    labels = ["Original GloVe", "WN_syn", "WN_all"]
    values = [result["original"], result["wn_syn"], result["wn_all"]]
    colors = ["#888888", "#444444", "#111111"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_title("SST-2: Original vs WN_syn vs WN_all")
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

    print("=== Building WN_syn graph ===")
    graph_syn = build_wordnet_graph(set(embeddings), include_synonyms=True,
                                    include_hypernyms=False, include_hyponyms=False)
    graph_syn = filter_graph_by_vocab(graph_syn, set(embeddings))
    retrofitted_syn, _ = retrofit_vectors(embeddings, graph_syn, num_iters=10, alpha=1.0)
    print("WN_syn done\n")

    print("=== Building WN_all graph ===")
    graph_all = build_wordnet_graph(set(embeddings), include_synonyms=True,
                                    include_hypernyms=True, include_hyponyms=True)
    graph_all = filter_graph_by_vocab(graph_all, set(embeddings))
    retrofitted_all, _ = retrofit_vectors(embeddings, graph_all, num_iters=10, alpha=1.0)
    print("WN_all done\n")

    result = evaluate_sst(embeddings, retrofitted_syn, retrofitted_all)
    plot_sst(result)