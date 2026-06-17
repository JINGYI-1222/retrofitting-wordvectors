from pathlib import Path
import csv
import sys
import time
import gc

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab
from src.eval import load_ws353, load_simlex, load_rg65, cosine_similarity


MAX_WORDS = 50000
NUM_ITERS = 10
BETA_STRATEGY = "inverse_degree"
ALPHAS = [0.5, 1.0, 2.0]

EMBEDDING_PATH = PROJECT_ROOT / "models/glove.6B.300d.txt"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "alpha_ablation_results.csv"
OUTPUT_LOG = RESULTS_DIR / "alpha_ablation_summary.txt"


def evaluate_with_coverage(vectors: dict, pairs: list) -> dict:
    """Evaluate word similarity and return rho plus coverage information."""
    human_scores = []
    model_scores = []
    skipped = 0

    for w1, w2, human_score in pairs:
        if w1 not in vectors or w2 not in vectors:
            skipped += 1
            continue

        human_scores.append(human_score)
        model_scores.append(cosine_similarity(vectors[w1], vectors[w2]))

    covered = len(human_scores)
    total = len(pairs)

    if covered < 2:
        rho = float("nan")
    else:
        rho, _ = spearmanr(human_scores, model_scores)
        rho = float(rho)

    return {
        "rho": rho,
        "covered": covered,
        "total": total,
        "skipped": skipped,
        "coverage": covered / total if total else 0.0,
    }


def assert_same_vocabulary(original_vectors: dict, retrofitted_vectors: dict) -> None:
    """Fail loudly if retrofitting changes the vocabulary."""
    if set(original_vectors) != set(retrofitted_vectors):
        missing = set(original_vectors) - set(retrofitted_vectors)
        extra = set(retrofitted_vectors) - set(original_vectors)
        raise AssertionError(
            f"Vocabulary mismatch. Missing={len(missing)}, extra={len(extra)}"
        )


def assert_all_vectors_finite(vectors: dict, label: str) -> None:
    """Fail loudly if any vector contains NaN or infinity."""
    for word, vector in vectors.items():
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} contains non-finite values for word: {word}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    log_lines = []

    def log(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    log("=== Alpha Ablation: GloVe + WordNet Synonym Retrofitting ===")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Embedding path: {EMBEDDING_PATH}")
    log(f"Max words: {MAX_WORDS}")
    log(f"Iterations: {NUM_ITERS}")
    log(f"Beta strategy: {BETA_STRATEGY}")
    log(f"Alpha values: {ALPHAS}")
    log()

    log("=== Loading embeddings ===")
    embeddings = load_text_embeddings(EMBEDDING_PATH, max_words=MAX_WORDS)
    assert_all_vectors_finite(embeddings, "Original embeddings")
    log(f"Loaded embeddings: {len(embeddings)}")
    first_vector = next(iter(embeddings.values()))
    log(f"Vector dimension: {len(first_vector)}")
    log()

    log("=== Building WordNet synonym graph ===")
    graph = build_wordnet_graph(set(embeddings), include_synonyms=True)
    graph = filter_graph_by_vocab(graph, set(embeddings))
    log(f"Filtered graph nodes: {len(graph)}")
    log()

    log("=== Loading evaluation datasets ===")
    datasets = {
        "WS-353": load_ws353(PROJECT_ROOT / "datasets/combined.csv"),
        "SimLex-999": load_simlex(PROJECT_ROOT / "datasets/SimLex-999.txt"),
        "RG-65": load_rg65(PROJECT_ROOT / "datasets/rg65.txt"),
    }
    log()

    log("=== Evaluating original embeddings once ===")
    original_results = {}
    for dataset_name, pairs in datasets.items():
        result = evaluate_with_coverage(embeddings, pairs)
        original_results[dataset_name] = result
        log(
            f"{dataset_name}: original rho={result['rho']:.4f}, "
            f"coverage={result['covered']}/{result['total']}, "
            f"skipped={result['skipped']}"
        )
    log()

    rows = []

    for alpha in ALPHAS:
        log(f"=== Running retrofitting for alpha={alpha} ===")
        start = time.perf_counter()

        retrofitted, stats = retrofit_vectors(
            embeddings,
            graph,
            num_iters=NUM_ITERS,
            alpha=alpha,
            beta_strategy=BETA_STRATEGY,
        )

        runtime_seconds = time.perf_counter() - start

        assert_same_vocabulary(embeddings, retrofitted)
        assert_all_vectors_finite(retrofitted, f"Retrofitted embeddings alpha={alpha}")

        log(f"Runtime seconds: {runtime_seconds:.4f}")
        log(f"Stats: {stats}")

        for dataset_name, pairs in datasets.items():
            before = original_results[dataset_name]
            after = evaluate_with_coverage(retrofitted, pairs)

            if before["covered"] != after["covered"]:
                raise AssertionError(
                    f"Coverage mismatch for {dataset_name} at alpha={alpha}: "
                    f"before={before['covered']}, after={after['covered']}"
                )

            delta = after["rho"] - before["rho"]

            row = {
                "experiment_type": "alpha_ablation",
                "embedding_source": "glove.6B.300d_first_50000",
                "graph": "wordnet_synonyms_filtered_by_vocab",
                "num_iters": NUM_ITERS,
                "alpha": alpha,
                "beta_strategy": BETA_STRATEGY,
                "dataset": dataset_name,
                "covered_pairs": after["covered"],
                "total_pairs": after["total"],
                "skipped_pairs": after["skipped"],
                "coverage": after["coverage"],
                "original_rho": before["rho"],
                "retrofitted_rho": after["rho"],
                "delta": delta,
                "runtime_seconds": runtime_seconds,
                "words_updated": stats.get("words_updated"),
                "words_unchanged": stats.get("words_unchanged"),
                "oov_neighbours_skipped": stats.get("oov_neighbours_skipped"),
            }
            rows.append(row)

            log(
                f"{dataset_name}: before={before['rho']:.4f}, "
                f"after={after['rho']:.4f}, delta={delta:+.4f}, "
                f"coverage={after['covered']}/{after['total']}"
            )

        log()

        del retrofitted
        gc.collect()

    fieldnames = [
        "experiment_type",
        "embedding_source",
        "graph",
        "num_iters",
        "alpha",
        "beta_strategy",
        "dataset",
        "covered_pairs",
        "total_pairs",
        "skipped_pairs",
        "coverage",
        "original_rho",
        "retrofitted_rho",
        "delta",
        "runtime_seconds",
        "words_updated",
        "words_unchanged",
        "oov_neighbours_skipped",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    log(f"Saved CSV results to: {OUTPUT_CSV}")
    log(f"Saved summary log to: {OUTPUT_LOG}")
    log("=== Alpha ablation completed successfully ===")


if __name__ == "__main__":
    main()
