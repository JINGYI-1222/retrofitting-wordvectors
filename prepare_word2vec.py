from pathlib import Path
import argparse

from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab, report_oov
from src.utils import load_word2vec_binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relations",
        choices=["syn", "all"],
        default="syn",
        help="WordNet relations to use: syn or all.",
    )
    parser.add_argument(
        "--embeddings",
        default="models/GoogleNews-vectors-negative300.bin.gz",
        help="Path to the binary Word2Vec 300d embedding file.",
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
        print("Word2Vec embedding file not found:")
        print(embedding_path)
        print()
        print("Expected setup:")
        print("models/GoogleNews-vectors-negative300.bin.gz")
        return

    print("Loading Word2Vec embeddings...")
    embeddings = load_word2vec_binary(embedding_path, max_words=args.max_words)
    vocab = set(embeddings)
    print("Loaded embeddings:", len(embeddings))

    include_hypernyms = args.relations == "all"
    include_hyponyms = args.relations == "all"
    graph_name = "WN_syn" if args.relations == "syn" else "WN_all"

    print(f"Building WordNet graph ({graph_name})...")
    raw_graph = build_wordnet_graph(
        vocab,
        include_synonyms=True,
        include_hypernyms=include_hypernyms,
        include_hyponyms=include_hyponyms,
        keep_only_vocab=False,
    )
    raw_stats = report_oov(raw_graph, embeddings)
    raw_edge_count = count_edges(raw_graph)

    print(f"Raw Word2Vec {graph_name} graph nodes:", len(raw_graph))
    print(f"Raw Word2Vec {graph_name} graph edges:", raw_edge_count)
    print("OOV before filtering:", raw_stats)

    print("Filtering graph by embedding vocabulary...")
    graph = filter_graph_by_vocab(raw_graph, vocab)
    stats = report_oov(graph, embeddings)
    edge_count = count_edges(graph)

    print(f"Filtered Word2Vec {graph_name} graph nodes:", len(graph))
    print(f"Filtered Word2Vec {graph_name} graph edges:", edge_count)
    print("OOV after filtering:", stats)

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
            print(f'graph["{example_word}"] = no WordNet neighbor kept in the graph')


def count_edges(graph):
    return sum(len(neighbors) for neighbors in graph.values()) // 2


if __name__ == "__main__":
    main()
