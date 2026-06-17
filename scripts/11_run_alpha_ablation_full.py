"""Run full-data alpha ablation for WordNet retrofitting."""

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
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab
from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings


DEFAULT_EMBEDDING_PATH = PROJECT_ROOT / "models/glove.6B.300d.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results/alpha_ablation_full.csv"
DEFAULT_ALPHAS = "0.5,1.0,2.0"
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
            "Test alpha sensitivity for retrofitting GloVe 6B 300d vectors "
            "with a WordNet graph."
        )
    )
    parser.add_argument(
        "--embedding-path",
        default=str(DEFAULT_EMBEDDING_PATH),
        help="Path to the GloVe 6B 300d text embedding file.",
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
        "--alphas",
        default=DEFAULT_ALPHAS,
        help='Comma-separated alpha values. Default: "0.5,1.0,2.0".',
    )
    parser.add_argument(
        "--beta-strategy",
        choices=["inverse_degree", "constant", "symmetric_degree"],
        default="inverse_degree",
        help="Beta weighting strategy to use during retrofitting.",
    )
    parser.add_argument(
        "--relations",
        default="syn",
        help='WordNet relations to use. Supported values: "syn" or "all".',
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
        help="CSV file to append alpha-condition results to.",
    )

    args = parser.parse_args()

    if args.max_words is not None and args.max_words <= 0:
        parser.error("--max-words must be positive when provided.")
    if args.num_iters < 0:
        parser.error("--num-iters must be non-negative.")

    args.alphas = parse_alpha_values(args.alphas, parser)
    args.graph_variant, args.graph_options = parse_relations(args.relations, parser)
    args.embedding_path = resolve_project_path(args.embedding_path)
    args.output = resolve_project_path(args.output)

    if not args.embedding_path.exists():
        parser.error(f"Embedding file does not exist: {args.embedding_path}")

    return args


def parse_alpha_values(alpha_text: str, parser: argparse.ArgumentParser) -> list[float]:
    try:
        alphas = [
            float(value.strip())
            for value in alpha_text.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error("--alphas must be a comma-separated list of numbers.")

    if not alphas:
        parser.error("--alphas must contain at least one value.")

    for alpha in alphas:
        if not np.isfinite(alpha) or alpha <= 0:
            parser.error("--alphas must contain finite values greater than zero.")

    return alphas


def parse_relations(
    relations: str,
    parser: argparse.ArgumentParser,
) -> tuple[str, dict[str, bool]]:
    normalized = relations.strip().lower()

    if normalized in {"syn", "synonym", "synonyms"}:
        return (
            "WN_syn",
            {
                "include_synonyms": True,
                "include_hypernyms": False,
                "include_hyponyms": False,
            },
        )

    if normalized == "all":
        return (
            "WN_all",
            {
                "include_synonyms": True,
                "include_hypernyms": True,
                "include_hyponyms": True,
            },
        )

    parser.error('--relations must be "syn" or "all".')


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

    print("=== Full-Data Alpha Ablation ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Embedding path: {args.embedding_path}")
    print(f"Output CSV: {args.output}")
    print(f"Max words: {max_words_label}")
    print(f"Graph variant: {args.graph_variant}")
    print(f"Beta strategy: {args.beta_strategy}")
    print(f"Alpha values: {args.alphas}")
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

    print(f"Building {args.graph_variant} graph...")
    graph = build_wordnet_graph(embedding_vocab, **args.graph_options)
    graph = filter_graph_by_vocab(graph, embedding_vocab)
    graph_nodes = len(graph)
    graph_edges = count_undirected_edges(graph)
    print(f"{args.graph_variant}: nodes={graph_nodes}, edges={graph_edges}")
    print()

    for alpha in args.alphas:
        print(f"Running {args.graph_variant} + {args.beta_strategy} + alpha={alpha}")
        condition_start = time.perf_counter()

        retrofitted_vectors, _stats = retrofit_vectors(
            embeddings,
            graph,
            num_iters=args.num_iters,
            alpha=alpha,
            beta_strategy=args.beta_strategy,
        )

        assert_same_vocabulary(embeddings, retrofitted_vectors)
        assert_all_vectors_finite(
            retrofitted_vectors,
            f"Retrofitted embeddings ({args.graph_variant}, alpha={alpha})",
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
                    "graph_variant": args.graph_variant,
                    "graph_nodes": graph_nodes,
                    "graph_edges": graph_edges,
                    "beta_strategy": args.beta_strategy,
                    "alpha": alpha,
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
        print(f"Saved rows for alpha={alpha}")
        print(f"Runtime seconds: {runtime_seconds:.2f}")
        print()

        del retrofitted_vectors
        del after_results
        del rows
        gc.collect()

    del graph
    gc.collect()

    print("Completed full-data alpha ablation.")


if __name__ == "__main__":
    main()
