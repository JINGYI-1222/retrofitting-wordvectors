import numpy as np

from src.retrofit import retrofit_vectors


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def make_mock_vectors():
    return {
        "good": np.array([1.0, 0.0]),
        "nice": np.array([0.8, 0.2]),
        "bad": np.array([-1.0, 0.0]),
    }


def test_good_and_nice_become_closer():
    original_vectors = make_mock_vectors()
    graph = {
        "good": ["nice"],
        "nice": ["good"],
        "bad": [],
    }

    retrofitted_vectors, stats = retrofit_vectors(
        original_vectors, graph, num_iters=1
    )

    original_similarity = cosine_similarity(
        original_vectors["good"], original_vectors["nice"]
    )
    retrofitted_similarity = cosine_similarity(
        retrofitted_vectors["good"], retrofitted_vectors["nice"]
    )

    assert retrofitted_similarity > original_similarity
    assert stats["words_updated"] == 2


def test_bad_remains_unchanged():
    original_vectors = make_mock_vectors()
    graph = {
        "good": ["nice"],
        "nice": ["good"],
        "bad": [],
    }

    retrofitted_vectors, stats = retrofit_vectors(
        original_vectors, graph, num_iters=1
    )

    np.testing.assert_allclose(retrofitted_vectors["bad"], original_vectors["bad"])
    assert stats["words_unchanged"] == 1


def test_original_vectors_are_not_modified_in_place():
    original_vectors = make_mock_vectors()
    original_copies = {
        word: vector.copy() for word, vector in original_vectors.items()
    }
    graph = {
        "good": ["nice"],
        "nice": ["good"],
        "bad": [],
    }

    retrofitted_vectors, _ = retrofit_vectors(original_vectors, graph, num_iters=1)

    for word in original_vectors:
        np.testing.assert_allclose(original_vectors[word], original_copies[word])
        assert not np.shares_memory(retrofitted_vectors[word], original_vectors[word])


def test_oov_neighbour_is_skipped():
    original_vectors = {
        "good": np.array([1.0, 0.0]),
    }
    graph = {
        "good": ["missing_word"],
    }

    retrofitted_vectors, stats = retrofit_vectors(
        original_vectors, graph, num_iters=1
    )

    np.testing.assert_allclose(retrofitted_vectors["good"], original_vectors["good"])
    assert stats["oov_neighbours_skipped"] == 1


def test_synchronous_update_is_used():
    original_vectors = {
        "a": np.array([0.0]),
        "b": np.array([10.0]),
    }
    graph = {
        "a": ["b"],
        "b": ["a"],
    }

    retrofitted_vectors, _ = retrofit_vectors(original_vectors, graph, num_iters=1)

    np.testing.assert_allclose(retrofitted_vectors["a"], np.array([5.0]))
    np.testing.assert_allclose(retrofitted_vectors["b"], np.array([5.0]))


def test_inverse_degree_beta_with_multiple_valid_neighbours():
    original_vectors = {
        "center": np.array([0.0]),
        "left": np.array([2.0]),
        "right": np.array([4.0]),
    }
    graph = {
        "center": ["left", "right"],
        "left": [],
        "right": [],
    }

    retrofitted_vectors, stats = retrofit_vectors(
        original_vectors,
        graph,
        num_iters=1,
        alpha=1.0,
        beta_strategy="inverse_degree",
    )

    np.testing.assert_allclose(retrofitted_vectors["center"], np.array([1.5]))
    assert stats["words_updated"] == 1


def test_accepts_set_neighbour_graph():
    original_vectors = {
        "good": np.array([1.0, 0.0]),
        "nice": np.array([0.8, 0.2]),
    }
    graph = {
        "good": {"nice"},
        "nice": {"good"},
    }

    retrofitted_vectors, stats = retrofit_vectors(
        original_vectors, graph, num_iters=1
    )

    assert set(retrofitted_vectors) == {"good", "nice"}
    assert stats["words_updated"] == 2
