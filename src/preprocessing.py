from collections import defaultdict
from pathlib import Path


def load_lexicon_edges(path: str | Path) -> dict[str, set[str]]:
    """Load a simple two-column lexicon file as an undirected graph."""
    graph = defaultdict(set)
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            word_a, word_b = line.split("\t")[:2]
            if word_a == word_b:
                continue

            graph[word_a].add(word_b)
            graph[word_b].add(word_a)

    return dict(graph)


def build_wordnet_graph(
    vocab: set[str],
    include_synonyms: bool = True,
    include_hypernyms: bool = False,
    include_hyponyms: bool = False,
) -> dict[str, set[str]]:
    """Build a WordNet graph, keeping only words that are in our vocabulary."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as exc:
        raise RuntimeError("NLTK is required to build a WordNet graph.") from exc

    graph = defaultdict(set)
    normalized_vocab = {word.lower() for word in vocab}

    for word in normalized_vocab:
        related_words = set()
        synsets = wn.synsets(word)

        for synset in synsets:
            if include_synonyms:
                related_words.update(_lemma_words(synset))

            if include_hypernyms:
                for hypernym in synset.hypernyms():
                    related_words.update(_lemma_words(hypernym))

            if include_hyponyms:
                for hyponym in synset.hyponyms():
                    related_words.update(_lemma_words(hyponym))

        for related_word in related_words:
            if related_word == word or related_word not in normalized_vocab:
                continue
            graph[word].add(related_word)
            graph[related_word].add(word)

    return dict(graph)


def _lemma_words(synset) -> set[str]:
    words = set()
    for lemma in synset.lemmas():
        name = lemma.name().lower()
        if "_" not in name:
            words.add(name)
    return words


def filter_graph_by_vocab(graph: dict[str, set[str]], vocab: set[str]) -> dict[str, set[str]]:
    """Remove nodes and edges that cannot be used because embeddings are missing."""
    filtered = {}

    for word, neighbors in graph.items():
        if word not in vocab:
            continue

        kept_neighbors = {neighbor for neighbor in neighbors if neighbor in vocab}
        if kept_neighbors:
            filtered[word] = kept_neighbors

    return filtered


def report_oov(graph: dict[str, set[str]], embeddings: dict[str, object]) -> dict[str, int]:
    """Count how many lexicon words are missing from the embedding vocabulary."""
    vocab = set(embeddings)
    graph_words = set(graph)
    for neighbors in graph.values():
        graph_words.update(neighbors)

    return {
        "embedding_vocab_size": len(vocab),
        "semantic_graph_vocab_size": len(graph_words),
        "oov_after_filtering": len(graph_words - vocab),
        "usable_graph_nodes": len(graph_words & vocab),
    }
