from pathlib import Path
import argparse

from src.preprocessing import build_ppdb_graph, filter_graph_by_vocab, report_oov
from src.utils import load_text_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppdb",
        default="datasets/ppdb/ppdb-xl.txt",
        help="Path to a PPDB lexical file. Plain text and .gz files are supported.",
    )
    parser.add_argument(
        "--embeddings",
        default="models/glove.6B.300d.txt",
        help="Path to the GloVe 300d embedding file.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional embedding limit for quick tests, for example 50000.",
    )
    args = parser.parse_args()

    ppdb_path = Path(args.ppdb)
    if not ppdb_path.exists():
        print("PPDB file not found:")
        print(ppdb_path)
        print()
        print("Expected setup:")
        print("datasets/ppdb/ppdb-xl.txt")
        print()
        print("Use a PPDB lexical file and pass its path with --ppdb if needed.")
        return

    embedding_path = Path(args.embeddings)
    if not embedding_path.exists():
        print("Embedding file not found:")
        print(embedding_path)
        print()
        print("Expected setup:")
        print("models/glove.6B.300d.txt")
        return

    print("Loading embeddings...")
    embeddings = load_text_embeddings(embedding_path, max_words=args.max_words)
    vocab = set(embeddings)
    print("Loaded embeddings:", len(embeddings))

    print("Building PPDB lexical graph...")
    raw_graph = build_ppdb_graph(ppdb_path, keep_only_vocab=False)
    raw_stats = report_oov(raw_graph, embeddings)
    raw_edge_count = count_edges(raw_graph)

    print("Raw PPDB graph nodes:", len(raw_graph))
    print("Raw PPDB graph edges:", raw_edge_count)
    print("OOV before filtering:", raw_stats)

    print("Filtering graph by embedding vocabulary...")
    graph = filter_graph_by_vocab(raw_graph, vocab)
    stats = report_oov(graph, embeddings)
    edge_count = count_edges(graph)

    print("Filtered PPDB graph nodes:", len(graph))
    print("Filtered PPDB graph edges:", edge_count)
    print("OOV after filtering:", stats)

    print()
    print("Data objects ready for retrofitting:")
    print("embeddings: dict[str, np.ndarray]")
    print("graph: dict[str, set[str]]")


def count_edges(graph):
    return sum(len(neighbors) for neighbors in graph.values()) // 2


if __name__ == "__main__":
    main()
