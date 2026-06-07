from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def euclidean_distance(vector_a, vector_b):
    return np.linalg.norm(vector_a - vector_b)


original_vectors = {
    "good": np.array([1.0, 0.0]),
    "nice": np.array([0.8, 0.2]),
    "bad": np.array([-1.0, 0.0]),
}

graph = {
    "good": ["nice"],
    "nice": ["good"],
    "bad": [],
}

retrofitted_vectors, stats = retrofit_vectors(original_vectors, graph)

print("original vectors:")
for word, vector in original_vectors.items():
    print(word, vector)

print()
print("retrofitted vectors:")
for word, vector in retrofitted_vectors.items():
    print(word, vector)

print()
print("stats:")
for name, value in stats.items():
    print(name, value)

print()
print(
    "original cosine(good, nice):",
    cosine_similarity(original_vectors["good"], original_vectors["nice"]),
)
print(
    "retrofitted cosine(good, nice):",
    cosine_similarity(retrofitted_vectors["good"], retrofitted_vectors["nice"]),
)
print(
    "original euclidean distance(good, nice):",
    euclidean_distance(original_vectors["good"], original_vectors["nice"]),
)
print(
    "retrofitted euclidean distance(good, nice):",
    euclidean_distance(retrofitted_vectors["good"], retrofitted_vectors["nice"]),
)
print(
    "bad stayed unchanged:",
    np.allclose(retrofitted_vectors["bad"], original_vectors["bad"]),
)
