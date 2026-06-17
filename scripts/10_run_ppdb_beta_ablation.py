"""Run PPDB beta-strategy ablations for English GloVe embeddings."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import cosine_similarity, load_rg65, load_simlex, load_ws353
from src.preprocessing import build_ppdb_graph, filter_graph_by_vocab
from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings


DEFAULT_EMBEDDING_PATH = PROJECT_ROOT / "models/glove.6B.300d.txt"
DEFAULT_PPDB_PATH = PROJECT_ROOT / "datasets/ppdb/ppdb-xl.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results/ppdb_beta_ablation_full.csv"
BETA_STRATEGIES = ["inverse_degree", "constant", "symmetric_degree"]
GRAPH_VARIANT = "PPDB"
FIELDNAMES = [
    "max_words",
    "embedding_vocab_size",
    "graph_variant",
    "graph_nodes",
    "graph_edges",
    "beta_strategy",
    "alpha",
    "num_iters",
    "dataset",
    "original_score",
    "retrofitted_score",
    "delta",
    "evaluated_pairs",
    "total_pairs",
    "coverage",
    "runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare beta strategies for retrofitting GloVe 6B 300d vectors "
            "with an English PPDB paraphrase graph."
        )
    )
    parser.add_argument(
        "--embedding-path",
        default=str(DEFAULT_EMBEDDING_PATH),
        help="Path to the GloVe 6B 300d text embedding file.",
    )
    parser.add_argument(
        "--ppdb-path",
        default=str(DEFAULT_PPDB_PATH),
        help="Path to the English PPDB text file.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help=(
            "Optional number of embedding rows to load. If omitted, the full "
            "GloVe vocabulary is loaded."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight assigned to each original vector.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of synchronous retrofitting iterations.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="CSV file to append condition results to.",
    )

    args = parser.parse_args()

    if args.max_words is not None and args.max_words <= 0:
        parser.error("--max-words must be positive when provided.")
    if args.num_iters < 0:
        parser.error("--num-iters must be non-negative.")
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and strictly greater than zero.")

    args.embedding_path = resolve_project_path(args.embedding_path)
    args.ppdb_path = resolve_project_path(args.ppdb_path)
    args.output = resolve_project_path(args.output)

    if not args.embedding_path.exists():
        parser.error(f"Embedding file does not exist: {args.embedding_path}")
    if not args.ppdb_path.exists():
        parser.error(f"PPDB file does not exist: {args.ppdb_path}")

    return args


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def evaluate_with_coverage(vectors: dict[str, np.ndarray], pairs: list) -> dict:
    human_scores = []
    model_scores = []

    for word_a, word_b, human_score in pairs:
        if word_a not in vectors or word_b not in vectors:
            continue

        human_scores.append(human_score)
        model_scores.append(cosine_similarity(vectors[word_a], vectors[word_b]))

    evaluated_pairs = len(human_scores)
    total_pairs = len(pairs)

    if evaluated_pairs < 2:
        score = float("nan")
    else:
        score, _ = spearmanr(human_scores, model_scores)
        score = float(score)

    return {
        "score": score,
        "evaluated_pairs": evaluated_pairs,
        "total_pairs": total_pairs,
        "coverage": evaluated_pairs / total_pairs if total_pairs else 0.0,
    }


def assert_same_vocabulary(
    original_vectors: dict[str, np.ndarray],
    retrofitted_vectors: dict[str, np.ndarray],
) -> None:
    if set(original_vectors) != set(retrofitted_vectors):
        missing = set(original_vectors) - set(retrofitted_vectors)
        extra = set(retrofitted_vectors) - set(original_vectors)
        raise AssertionError(
            f"Vocabulary mismatch. Missing={len(missing)}, extra={len(extra)}"
        )


def assert_all_vectors_finite(
    vectors: dict[str, np.ndarray],
    label: str,
) -> None:
    for word, vector in vectors.items():
        if not np.all(np.isfinite(vector)):
            raise AssertionError(f"{label} contains non-finite values for {word}")


def count_undirected_edges(graph: dict[str, set[str]]) -> int:
    return sum(len(neighbours) for neighbours in graph.values()) // 2


def append_rows(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_evaluation_datasets() -> dict[str, list]:
    return {
        "WS-353": load_ws353(PROJECT_ROOT / "datasets/combined.csv"),
        "SimLex-999": load_simlex(PROJECT_ROOT / "datasets/SimLex-999.txt"),
        "RG-65": load_rg65(PROJECT_ROOT / "datasets/rg65.txt"),
    }


def print_dataset_result(dataset_name: str, before: dict, after: dict) -> None:
    delta = after["score"] - before["score"]
    print(
        f"  {dataset_name}: original={before['score']:.4f}, "
        f"retrofitted={after['score']:.4f}, delta={delta:+.4f}, "
        f"coverage={after['evaluated_pairs']}/{after['total_pairs']}"
    )


def main() -> None:
    args = parse_args()
    max_words_label = args.max_words if args.max_words is not None else "full"

    print("=== PPDB Beta Ablation ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Embedding path: {args.embedding_path}")
    print(f"PPDB path: {args.ppdb_path}")
    print(f"Output CSV: {args.output}")
    print(f"Max words: {max_words_label}")
    print(f"Alpha: {args.alpha}")
    print(f"Iterations: {args.num_iters}")
    print()

    print("Loading embeddings...")
    embeddings = load_text_embeddings(args.embedding_path, max_words=args.max_words)
    if not embeddings:
        raise RuntimeError("No embeddings were loaded.")
    assert_all_vectors_finite(embeddings, "Original embeddings")
    embedding_vocab = set(embeddings)
    embedding_vocab_size = len(embeddings)
    print(f"Loaded embeddings: {embedding_vocab_size}")
    print(f"Vector dimension: {len(next(iter(embeddings.values())))}")
    print()

    print("Loading evaluation datasets...")
    datasets = load_evaluation_datasets()
    print()

    print("Evaluating original embeddings once...")
    original_results = {}
    for dataset_name, pairs in datasets.items():
        result = evaluate_with_coverage(embeddings, pairs)
        original_results[dataset_name] = result
        print(
            f"  {dataset_name}: score={result['score']:.4f}, "
            f"coverage={result['evaluated_pairs']}/{result['total_pairs']}"
        )
    print()

    for beta_strategy in BETA_STRATEGIES:
        print(f"Running {GRAPH_VARIANT} + {beta_strategy}")
        condition_start = time.perf_counter()

        print("  Building and filtering PPDB graph...")
        graph = build_ppdb_graph(
            args.ppdb_path,
            keep_only_vocab=True,
            vocab=embedding_vocab,
        )
        graph = filter_graph_by_vocab(graph, embedding_vocab)
        graph_nodes = len(graph)
        graph_edges = count_undirected_edges(graph)
        print(f"  Graph nodes: {graph_nodes}")
        print(f"  Graph edges: {graph_edges}")

        print("  Running retrofitting...")
        retrofitted_vectors, _stats = retrofit_vectors(
            embeddings,
            graph,
            num_iters=args.num_iters,
            alpha=args.alpha,
            beta_strategy=beta_strategy,
        )

        assert_same_vocabulary(embeddings, retrofitted_vectors)
        assert_all_vectors_finite(
            retrofitted_vectors,
            f"Retrofitted embeddings ({GRAPH_VARIANT}, {beta_strategy})",
        )

        after_results = {}
        for dataset_name, pairs in datasets.items():
            before = original_results[dataset_name]
            after = evaluate_with_coverage(retrofitted_vectors, pairs)
            after_results[dataset_name] = after

            if before["evaluated_pairs"] != after["evaluated_pairs"]:
                raise AssertionError(
                    f"Coverage mismatch for {dataset_name}: "
                    f"original={before['evaluated_pairs']}, "
                    f"retrofitted={after['evaluated_pairs']}"
                )

        runtime_seconds = time.perf_counter() - condition_start

        rows = []
        for dataset_name, after in after_results.items():
            before = original_results[dataset_name]
            delta = after["score"] - before["score"]
            rows.append(
                {
                    "max_words": max_words_label,
                    "embedding_vocab_size": embedding_vocab_size,
                    "graph_variant": GRAPH_VARIANT,
                    "graph_nodes": graph_nodes,
                    "graph_edges": graph_edges,
                    "beta_strategy": beta_strategy,
                    "alpha": args.alpha,
                    "num_iters": args.num_iters,
                    "dataset": dataset_name,
                    "original_score": before["score"],
                    "retrofitted_score": after["score"],
                    "delta": delta,
                    "evaluated_pairs": after["evaluated_pairs"],
                    "total_pairs": after["total_pairs"],
                    "coverage": after["coverage"],
                    "runtime_seconds": runtime_seconds,
                }
            )
            print_dataset_result(dataset_name, before, after)

        append_rows(args.output, rows)
        print(f"Saved rows for {GRAPH_VARIANT} + {beta_strategy}")
        print(f"Runtime seconds: {runtime_seconds:.2f}")
        print()

        del retrofitted_vectors
        del after_results
        del rows
        del graph
        gc.collect()

    print("Completed PPDB beta ablation.")


if __name__ == "__main__":
    main()
