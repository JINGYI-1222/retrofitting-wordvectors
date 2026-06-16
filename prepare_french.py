from pathlib import Path
import argparse

from src.preprocessing import build_wolf_graph, filter_graph_by_vocab, report_oov
from src.utils import load_text_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relations",
        choices=["syn", "all"],
        default="syn",
        help="WOLF relations to use: syn or all.",
    )
    parser.add_argument(
        "--embeddings",
        default="models/cc.fr.300.vec",
        help="Path to the French fastText .vec file.",
    )
    parser.add_argument(
        "--wolf",
        default="datasets/wolf/wolf-1.0b4.xml",
        help="Path to the WOLF XML file.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional limit for quick tests, for example 50000.",
    )
    args = parser.parse_args()

    embedding_path = Path(args.embeddings)
    wolf_path = Path(args.wolf)

    if not embedding_path.exists() or not wolf_path.exists():
        print("French resource file not found.")
        print()
        print("Expected setup:")
        print("models/cc.fr.300.vec")
        print("datasets/wolf/wolf-1.0b4.xml")
        print()
        print("You can prepare them with:")
        print("python3 download_french_resources.py")
        return

    include_hypernyms = args.relations == "all"
    include_hyponyms = args.relations == "all"
    graph_name = "WOLF_syn" if args.relations == "syn" else "WOLF_all"

    print("Loading French embeddings...")
    embeddings = load_text_embeddings(embedding_path, max_words=args.max_words)
    vocab = set(embeddings)
    print("Loaded embeddings:", len(embeddings))

    print(f"Building French WOLF graph ({graph_name})...")
    raw_graph = build_wolf_graph(
        wolf_path,
        include_synonyms=True,
        include_hypernyms=include_hypernyms,
        include_hyponyms=include_hyponyms,
        keep_only_vocab=False,
    )
    raw_stats = report_oov(raw_graph, embeddings)
    raw_edge_count = count_edges(raw_graph)

    print(f"Raw {graph_name} graph nodes:", len(raw_graph))
    print(f"Raw {graph_name} graph edges:", raw_edge_count)
    print("OOV before filtering:", raw_stats)

    print("Filtering graph by embedding vocabulary...")
    graph = filter_graph_by_vocab(raw_graph, vocab)
    stats = report_oov(graph, embeddings)
    edge_count = count_edges(graph)

    print(f"Filtered {graph_name} graph nodes:", len(graph))
    print(f"Filtered {graph_name} graph edges:", edge_count)
    print("OOV after filtering:", stats)

    print()
    print("French data objects ready for retrofitting:")
    print("embeddings: dict[str, np.ndarray]")
    print("graph: dict[str, set[str]]")

    example_word = "voiture"
    if example_word in embeddings:
        print()
        print("Example:")
        print(f'embeddings["{example_word}"].shape = {embeddings[example_word].shape}')
        if example_word in graph:
            print(f'graph["{example_word}"] first neighbors = {sorted(graph[example_word])[:10]}')
        else:
            print(f'graph["{example_word}"] = no WOLF neighbor kept in the graph')


def count_edges(graph):
    return sum(len(neighbors) for neighbors in graph.values()) // 2


if __name__ == "__main__":
    main()
