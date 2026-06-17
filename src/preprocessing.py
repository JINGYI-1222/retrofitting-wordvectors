from collections import defaultdict
from pathlib import Path
import gzip
import xml.etree.ElementTree as ET


def build_ppdb_graph(path: str | Path, keep_only_vocab: bool = False, vocab: set[str] | None = None) -> dict[str, set[str]]:
    """Build a graph from PPDB lexical paraphrases."""
    graph = defaultdict(set)
    normalized_vocab = {word.lower() for word in vocab} if vocab is not None else None

    with _open_text(path) as handle:
        for line in handle:
            words = _parse_ppdb_words(line)
            if len(words) < 2:
                continue

            _add_word_edges(graph, words, keep_only_vocab, normalized_vocab)

    return dict(graph)


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _parse_ppdb_words(line: str) -> set[str]:
    parts = [part.strip() for part in line.strip().split("|||")]
    if len(parts) >= 3:
        words = [_clean_ppdb_word(parts[1]), _clean_ppdb_word(parts[2])]
    else:
        parts = line.strip().split()
        words = [_clean_ppdb_word(part) for part in parts]

    return {word for word in words if word is not None}


def _clean_ppdb_word(word: str) -> str | None:
    word = word.strip().lower()
    if not word:
        return None
    if " " in word or "_" in word:
        return None
    if any(char.isdigit() for char in word):
        return None
    return word


def build_wordnet_graph(
    vocab: set[str],
    include_synonyms: bool = True,
    include_hypernyms: bool = False,
    include_hyponyms: bool = False,
    keep_only_vocab: bool = True,
) -> dict[str, set[str]]:
    """Build a WordNet graph from words in the embedding vocabulary."""
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
            if related_word == word:
                continue
            if keep_only_vocab and related_word not in normalized_vocab:
                continue
            graph[word].add(related_word)
            graph[related_word].add(word)

    return dict(graph)


def build_wolf_graph(
    path: str | Path,
    include_synonyms: bool = True,
    include_hypernyms: bool = False,
    include_hyponyms: bool = False,
    keep_only_vocab: bool = True,
    vocab: set[str] | None = None,
) -> dict[str, set[str]]:
    """Build a French semantic graph from WOLF."""
    normalized_vocab = {word.lower() for word in vocab} if vocab is not None else None
    synset_words = {}
    hypernym_links = []

    tree = ET.parse(path)
    root = tree.getroot()

    for synset in root.findall("SYNSET"):
        synset_id = synset.findtext("ID")
        words = _wolf_synset_words(synset)
        if not synset_id or not words:
            continue

        synset_words[synset_id] = words

        for relation in synset.findall("ILR"):
            if relation.get("type") == "hypernym" and relation.text:
                hypernym_links.append((synset_id, relation.text.strip()))

    graph = defaultdict(set)

    if include_synonyms:
        for words in synset_words.values():
            _add_word_edges(graph, words, keep_only_vocab, normalized_vocab)

    if include_hypernyms or include_hyponyms:
        for source_id, target_id in hypernym_links:
            source_words = synset_words.get(source_id, set())
            target_words = synset_words.get(target_id, set())
            for word_a in source_words:
                for word_b in target_words:
                    _add_one_edge(graph, word_a, word_b, keep_only_vocab, normalized_vocab)

    return dict(graph)


def _wolf_synset_words(synset) -> set[str]:
    words = set()
    synonym = synset.find("SYNONYM")
    if synonym is None:
        return words

    for literal in synonym.findall("LITERAL"):
        if literal.text is None:
            continue

        word = literal.text.strip().lower()
        if _is_simple_word(word):
            words.add(word)

    return words


def _lemma_words(synset) -> set[str]:
    words = set()
    for lemma in synset.lemmas():
        name = lemma.name().lower()
        if _is_simple_word(name):
            words.add(name)
    return words


def _add_word_edges(
    graph: dict[str, set[str]],
    words: set[str],
    keep_only_vocab: bool,
    vocab: set[str] | None,
) -> None:
    word_list = sorted(words)
    for index, word_a in enumerate(word_list):
        for word_b in word_list[index + 1:]:
            _add_one_edge(graph, word_a, word_b, keep_only_vocab, vocab)


def _add_one_edge(
    graph: dict[str, set[str]],
    word_a: str,
    word_b: str,
    keep_only_vocab: bool,
    vocab: set[str] | None,
) -> None:
    if word_a == word_b:
        return
    if keep_only_vocab and vocab is not None:
        if word_a not in vocab or word_b not in vocab:
            return

    graph[word_a].add(word_b)
    graph[word_b].add(word_a)


def _is_simple_word(word: str) -> bool:
    if not word or word == "_empty_":
        return False
    if " " in word or "_" in word:
        return False
    return True


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
        "oov_words": len(graph_words - vocab),
        "usable_graph_nodes": len(graph_words & vocab),
    }
