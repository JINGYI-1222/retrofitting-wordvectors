import argparse
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab
from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings


PREFERRED_SAMPLE_WORDS = ["car", "good", "bad", "dog", "cat"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run real WordNet synonym retrofitting on a configurable "
            "GloVe vocabulary."
        )
    )
    parser.add_argument(
        "--embedding-path",
        default="models/glove.6B.300d.txt",
        help="Path to a text embedding file.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=1000,
        help="Maximum number of embedding rows to load.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of retrofitting iterations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for the original vector.",
    )

    args = parser.parse_args()

    if args.max_words <= 0:
        parser.error("--max-words must be positive.")

    if args.num_iters < 0:
        parser.error("--num-iters must be non-negative.")

    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and strictly greater than zero.")

    args.embedding_path = resolve_project_path(args.embedding_path)
    if not args.embedding_path.exists():
        parser.error(f"Embedding file does not exist: {args.embedding_path}")

    return args


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def vector_dimension(embeddings):
    first_vector = next(iter(embeddings.values()), None)
    if first_vector is None:
        raise RuntimeError("No embeddings were loaded.")
    return len(first_vector)


def verify_all_vectors_finite(vectors, label):
    for word, vector in vectors.items():
        if not np.isfinite(vector).all():
            raise RuntimeError(
                f"{label} vector contains NaN or Inf for word: {word}"
            )


def count_undirected_edges(graph):
    # Dividing the adjacency total by two assumes preprocessing stores every
    # undirected relation in both directions.
    return sum(len(neighbours) for neighbours in graph.values()) // 2


def select_sample_words(embeddings, graph):
    sample_words = [
        word for word in PREFERRED_SAMPLE_WORDS if word in embeddings
    ][:3]

    if len(sample_words) < 3:
        for word in sorted(graph):
            if len(sample_words) >= 3:
                break
            if word in embeddings and word not in sample_words and graph[word]:
                sample_words.append(word)

    return sample_words


def verify_retrofit_output(embeddings, retrofitted_vectors, sample_copies):
    if len(retrofitted_vectors) != len(embeddings):
        raise RuntimeError(
            "Retrofitted vector count does not match embedding count."
        )

    if retrofitted_vectors.keys() != embeddings.keys():
        raise RuntimeError(
            "Retrofitted vocabulary does not exactly match input vocabulary."
        )

    for word, original_copy in sample_copies.items():
        if not np.array_equal(embeddings[word], original_copy):
            raise RuntimeError(
                f"Original embedding was modified for sample word: {word}"
            )

    for word, retrofitted_vector in retrofitted_vectors.items():
        if retrofitted_vector.shape != embeddings[word].shape:
            raise RuntimeError(
                f"Vector dimension mismatch for word: {word}"
            )


def print_sample_diagnostics(embeddings, graph, retrofitted_vectors, sample_words):
    print()
    print("sample diagnostics:")

    if not sample_words:
        print("No preferred sample words found in the loaded embeddings.")
        return

    for word in sample_words:
        neighbours = sorted(graph.get(word, set()))
        has_valid_neighbours = bool(neighbours)
        euclidean_change = np.linalg.norm(
            retrofitted_vectors[word] - embeddings[word]
        )

        print(f"word: {word}")
        print(f"has valid graph neighbours: {has_valid_neighbours}")
        print(f"up to 5 sorted graph neighbours: {neighbours[:5]}")
        print(f"original L2 norm: {np.linalg.norm(embeddings[word])}")
        print(
            f"retrofitted L2 norm: {np.linalg.norm(retrofitted_vectors[word])}"
        )
        print(f"euclidean change: {euclidean_change}")


def main():
    args = parse_args()
    start_time = time.perf_counter()
    preparation_start = time.perf_counter()

    embeddings = load_text_embeddings(
        args.embedding_path,
        max_words=args.max_words,
    )
    dimension = vector_dimension(embeddings)
    verify_all_vectors_finite(embeddings, "input embeddings")

    graph = build_wordnet_graph(
        set(embeddings),
        include_synonyms=True,
        include_hypernyms=False,
        include_hyponyms=False,
    )
    graph = filter_graph_by_vocab(graph, set(embeddings))

    sample_words = select_sample_words(embeddings, graph)
    sample_copies = {
        word: embeddings[word].copy() for word in sample_words
    }

    preparation_seconds = time.perf_counter() - preparation_start

    retrofit_start = time.perf_counter()
    retrofitted_vectors, stats = retrofit_vectors(
        embeddings,
        graph,
        num_iters=args.num_iters,
        alpha=args.alpha,
        beta_strategy="inverse_degree",
    )
    retrofitting_seconds = time.perf_counter() - retrofit_start

    verify_all_vectors_finite(retrofitted_vectors, "retrofitted output")
    verify_retrofit_output(embeddings, retrofitted_vectors, sample_copies)

    elapsed_seconds = time.perf_counter() - start_time

    print("number of loaded embeddings:", len(embeddings))
    print("vector dimension:", dimension)
    print("number of filtered graph nodes:", len(graph))
    print("number of undirected graph edges:", count_undirected_edges(graph))
    print("number of iterations:", args.num_iters)
    print("alpha:", args.alpha)
    print("preparation seconds:", preparation_seconds)
    print("retrofitting seconds:", retrofitting_seconds)

    print()
    print("retrofit stats:")
    for name, value in stats.items():
        print(name, value)

    print()
    print("verification:")
    print("retrofitted vector count check: passed")
    print("retrofitted vocabulary check: passed")
    print("input finite-value check: passed")
    print("retrofitted finite-value check: passed")
    print("All sampled original vectors unchanged: True")
    print("vector dimension check: passed")

    print_sample_diagnostics(
        embeddings,
        graph,
        retrofitted_vectors,
        sample_words,
    )

    print()
    print("elapsed seconds:", elapsed_seconds)


if __name__ == "__main__":
    main()
