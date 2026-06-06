from pathlib import Path
import argparse

from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab, report_oov
from src.utils import load_text_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        default="models/glove.6B.300d.txt",
        help="Path to the GloVe 300d embedding file.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional limit for quick tests, for example 50000.",
    )
    args = parser.parse_args()

    embedding_path = Path(args.embeddings)
    if not embedding_path.exists():
        print("Embedding file not found:")
        print(embedding_path)
        print()
        print("Expected first version setup:")
        print("models/glove.6B.300d.txt")
        return

    print("Loading embeddings...")
    embeddings = load_text_embeddings(embedding_path, max_words=args.max_words)
    vocab = set(embeddings)
    print("Loaded embeddings:", len(embeddings))

    print("Building WordNet synonym graph...")
    graph = build_wordnet_graph(
        vocab,
        include_synonyms=True,
        include_hypernyms=False,
        include_hyponyms=False,
    )

    print("Filtering graph by embedding vocabulary...")
    graph = filter_graph_by_vocab(graph, vocab)
    stats = report_oov(graph, embeddings)

    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    print("WN_syn graph nodes:", len(graph))
    print("WN_syn graph edges:", edge_count)
    print("OOV report:", stats)

    print()
    print("Data objects ready for retrofitting:")
    print("embeddings: dict[str, np.ndarray]")
    print("graph: dict[str, set[str]]")

    example_word = "car"
    if example_word in embeddings:
        print()
        print("Example:")
        print(f'embeddings["{example_word}"].shape = {embeddings[example_word].shape}')
        if example_word in graph:
            print(f'graph["{example_word}"] first neighbors = {sorted(graph[example_word])[:10]}')
        else:
            print(f'graph["{example_word}"] = no WordNet synonym kept in the graph')


if __name__ == "__main__":
    main()
