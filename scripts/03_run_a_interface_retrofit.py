from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import filter_graph_by_vocab, load_lexicon_edges
from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings


MOCK_EMBEDDINGS_PATH = PROJECT_ROOT / "datasets/mock/mock_embeddings.txt"
MOCK_LEXICON_PATH = PROJECT_ROOT / "datasets/mock/mock_lexicon.tsv"


def print_sample_vectors(title, vectors, sample_words):
    print(title)
    for word in sample_words:
        print(word, vectors[word])


embeddings = load_text_embeddings(MOCK_EMBEDDINGS_PATH)
graph = load_lexicon_edges(MOCK_LEXICON_PATH)
graph = filter_graph_by_vocab(graph, set(embeddings))

retrofitted_vectors, stats = retrofit_vectors(embeddings, graph)

sample_words = list(embeddings)[:3]

print("number of embeddings:", len(embeddings))
print("number of graph entries:", len(graph))
print("number of retrofitted vectors:", len(retrofitted_vectors))

print()
print("stats:")
for name, value in stats.items():
    print(name, value)

print()
print_sample_vectors("sample original vectors:", embeddings, sample_words)

print()
print_sample_vectors(
    "sample retrofitted vectors:", retrofitted_vectors, sample_words
)
