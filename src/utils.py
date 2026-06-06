from pathlib import Path

import numpy as np


def load_text_embeddings(path: str | Path, max_words: int | None = None) -> dict[str, np.ndarray]:
    """Load a text embedding file, for example a GloVe file."""
    embeddings = {}
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float64)
            embeddings[word] = vector

            if max_words is not None and len(embeddings) >= max_words:
                break

    return embeddings


def save_text_embeddings(embeddings: dict[str, np.ndarray], path: str | Path) -> None:
    """Save embeddings as: word value1 value2 ..."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for word, vector in embeddings.items():
            values = " ".join(f"{value:.8f}" for value in vector)
            handle.write(f"{word} {values}\n")


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
