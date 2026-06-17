"""Core retrofitting algorithm for word vectors."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np


def retrofit_vectors(
    original_vectors: dict[str, np.ndarray],
    graph: dict[str, Collection[str]],
    num_iters: int = 10,
    alpha: float = 1.0,
    beta_strategy: str = "inverse_degree",
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Retrofit word vectors using a semantic neighbour graph.

    The update is synchronous: each iteration reads neighbour vectors from
    the previous iteration and writes updated vectors into a fresh dictionary.
    """
    if beta_strategy not in {
        "inverse_degree",
        "constant",
        "symmetric_degree",
    }:
        raise ValueError(
            "beta_strategy must be 'inverse_degree', 'constant', "
            "or 'symmetric_degree'."
        )

    valid_neighbours_by_word = {}
    oov_neighbours_skipped = 0

    for word in original_vectors:
        neighbours = graph.get(word, [])
        valid_neighbours = []

        for neighbour in neighbours:
            if neighbour in original_vectors:
                valid_neighbours.append(neighbour)
            else:
                oov_neighbours_skipped += 1

        valid_neighbours_by_word[word] = valid_neighbours

    words_with_no_valid_neighbours = sum(
        1 for neighbours in valid_neighbours_by_word.values() if not neighbours
    )
    if num_iters > 0:
        words_updated = len(original_vectors) - words_with_no_valid_neighbours
        words_unchanged = words_with_no_valid_neighbours
    else:
        words_updated = 0
        words_unchanged = len(original_vectors)

    stats = {
        "oov_neighbours_skipped": oov_neighbours_skipped,
        "words_with_no_valid_neighbours": words_with_no_valid_neighbours,
        "words_updated": words_updated,
        "words_unchanged": words_unchanged,
    }

    current_vectors = {
        word: vector.copy() for word, vector in original_vectors.items()
    }

    for _ in range(num_iters):
        previous_vectors = {
            word: vector.copy() for word, vector in current_vectors.items()
        }
        next_vectors = {}

        for word, original_vector in original_vectors.items():
            valid_neighbours = valid_neighbours_by_word[word]
            if not valid_neighbours:
                next_vectors[word] = original_vector.copy()
                continue

            neighbour_sum = np.zeros_like(original_vector, dtype=float)
            beta_sum = 0.0
            degree_i = max(1, len(valid_neighbours))

            for neighbour in valid_neighbours:
                if beta_strategy == "inverse_degree":
                    beta = 1.0 / len(valid_neighbours)
                elif beta_strategy == "constant":
                    beta = 1.0
                else:
                    degree_j = max(1, len(valid_neighbours_by_word[neighbour]))
                    beta = 1.0 / np.sqrt(degree_i * degree_j)

                neighbour_sum += beta * previous_vectors[neighbour]
                beta_sum += beta

            denominator = alpha + beta_sum
            next_vectors[word] = (
                alpha * original_vector + neighbour_sum
            ) / denominator

        current_vectors = next_vectors

    return current_vectors, stats
