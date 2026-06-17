# Retrofitting Word Vectors

This project reproduces the retrofitting method from Faruqui et al. (2015).

## Data processing Structure

```text
project/
├── README.md                         # project overview and running commands
│
├── src/
│   ├── utils.py                      # load/save embeddings, cosine similarity
│   └── preprocessing.py              # build semantic graphs and filter OOV words
│
├── prepare_wordnet.py                # English GloVe + WordNet
├── prepare_PPDB.py                   # English GloVe + PPDB
├── prepare_french.py                 # French fastText + WOLF
├── download_french_resources.py      # download French fastText and WOLF
│
├── models/
│   ├── glove.6B.300d.txt             # English pretrained embeddings
│   └── cc.fr.300.vec                 # French pretrained embeddings
│
└── datasets/
    ├── ppdb/
    │   └── ppdb-xl.txt               # English paraphrase lexicon
    └── wolf/
        └── wolf-1.0b4.xml            # French WordNet lexical resource
```

`models/` contains pretrained word embeddings.

`datasets/` contains semantic lexical resources.

## Data Loading and Preprocessing

This part prepares the data objects used by the retrofitting algorithm:

```python
embeddings: dict[str, np.ndarray]
graph: dict[str, set[str]]
```

`embeddings` stores the original pretrained word vectors.

`graph` stores the semantic neighbors from WordNet, WOLF, or PPDB.

## Current Data Combinations

English:

```text
GloVe 6B 300d + English WordNet synonyms
GloVe 6B 300d + English WordNet synonyms + hypernyms + hyponyms
GloVe 6B 300d + PPDB
```

French:

```text
French fastText 300d + WOLF synonyms
French fastText 300d + WOLF synonyms + hypernym links
```

## Files

```text
src/utils.py                  # load/save embeddings, cosine similarity
src/preprocessing.py          # build semantic graphs and filter OOV words

prepare_wordnet.py            # English GloVe + WordNet
prepare_PPDB.py               # English GloVe + PPDB
download_french_resources.py  # download French fastText + WOLF
prepare_french.py             # French fastText + WOLF
```

## Data Sources

English GloVe:

[Stanford GloVe project page](https://nlp.stanford.edu/projects/glove/)

Use:

```text
Wikipedia 2014 + Gigaword 5
glove.6B.300d.txt
```

Place it at:

```text
models/glove.6B.300d.txt
```

French fastText:

[fastText word vectors for 157 languages](https://fasttext.cc/docs/en/crawl-vectors.html)

The download script prepares:

```text
models/cc.fr.300.vec
```

French WOLF:

[WOLF French WordNet](https://almanach.inria.fr/software_and_resources/downloads/wolf-1.0b4.xml.bz2)

The download script prepares:

```text
datasets/wolf/wolf-1.0b4.xml
```

PPDB:

Place the PPDB file at:

```text
datasets/ppdb/ppdb-xl.txt
```

Folder convention:

```text
models/    pretrained word embeddings
datasets/  semantic lexical resources
```

Large resource files are not uploaded to GitHub.

## OOV Handling

The same OOV strategy is used for WordNet, WOLF, and PPDB.

Some words appear in a semantic lexicon but do not have a pretrained vector. Since retrofitting needs an initial vector for each updated word, these words cannot be directly updated.

We therefore:

- build the raw semantic graph;
- keep only words that appear in the embedding vocabulary;
- remove edges connected to words without pretrained vectors;
- remove graph nodes that have no remaining neighbors.

This gives a usable graph aligned with the embedding vocabulary.

## Run

English WordNet synonyms:

```bash
python3 prepare_wordnet.py --relations syn
```

English WordNet all:

```bash
python3 prepare_wordnet.py --relations all
```

English PPDB:

```bash
python3 prepare_PPDB.py
```

Download French resources:

```bash
python3 download_french_resources.py
```

French WOLF synonyms:

```bash
python3 prepare_french.py --relations syn
```

French WOLF all:

```bash
python3 prepare_french.py --relations all
```

(option)For quick tests, add:

```bash
--max-words 50000
```

----------------------------07/06/2025_update------------------------------

## Core Retrofitting Implementation

The current pipeline uses:

- GloVe 6B 300-dimensional English embeddings;
- an English WordNet graph;
- synonym relations only for the current baseline;
- vocabulary filtering before retrofitting;
- synchronous iterative vector updates.

### Team Responsibilities

- Person A: loads GloVe embeddings, constructs the WordNet graph, and filters the graph by the embedding vocabulary.
- Person B: implements and verifies the core retrofitting algorithm.
- Person C / the group: conducts later evaluation and comparison between original and retrofitted embeddings.

### Person B Interface

python from src.retrofit import retrofit_vectors  retrofitted_vectors, stats = retrofit_vectors(     original_vectors,     graph,     num_iters=10,     alpha=1.0,     beta_strategy="inverse_degree", ) 

Expected inputs:

python original_vectors: dict[str, np.ndarray] graph: dict[str, Collection[str]] 

Returned outputs:

python retrofitted_vectors: dict[str, np.ndarray] stats: dict[str, int] 

The returned statistics include:

- oov_neighbours_skipped
- words_with_no_valid_neighbours
- words_updated
- words_unchanged

### Implementation Properties

The current implementation:

- uses synchronous updates;
- reads neighbour vectors only from the previous iteration;
- does not modify the original input vectors in place;
- skips out-of-vocabulary neighbours;
- leaves words without valid neighbours unchanged;
- supports neighbour collections such as set[str];
- uses inverse-degree neighbour weighting;
- preserves the complete input vocabulary and vector dimensions.

### Tests

Run the test suite from the project root:

bash cd ~/Desktop/nlp-retrofitting-project .venv/bin/python -m pytest 

Current verified result:

text 7 passed 

The tests cover:

1. semantic neighbours become closer;
2. isolated words remain unchanged;
3. original vectors are not modified;
4. out-of-vocabulary neighbours are skipped;
5. updates are synchronous;
6. inverse-degree weighting works with multiple neighbours;
7. graph neighbour values may be stored as sets.

### Real-Data Integration

The real-data integration runner is:

text scripts/04_run_wn_syn_retrofit.py 

Example:

bash .venv/bin/python scripts/04_run_wn_syn_retrofit.py \   --max-words 1000 \   --num-iters 1 

The script:

- loads a configurable prefix of the GloVe file;
- builds a WordNet synonym graph;
- filters the graph by the embedding vocabulary;
- calls the core retrofitting implementation;
- verifies vocabulary and dimensional consistency;
- checks that sampled original vectors were not modified;
- checks that all input and output vectors contain only finite values;
- prints deterministic sample diagnostics and runtime information.

### Verified Configurations

The following configurations have completed successfully:

| Vocabulary | Iterations | Graph nodes | Undirected edges | Result |
|---:|---:|---:|---:|---|
| 1,000 | 1 | 595 | 1,232 | Passed |
| 1,000 | 10 | 595 | 1,232 | Passed |
| 50,000 | 1 | 25,616 | 80,232 | Passed |
| 50,000 | 10 | 25,616 | 80,232 | Passed |

For the 50,000-word, 10-iteration run:

text words updated: 25,616 words unchanged: 24,384 all input vectors finite: passed all output vectors finite: passed peak memory: approximately 1.10 GB 

These runs verify implementation correctness, numerical stability, and scalability at the tested sizes. They do not yet demonstrate that retrofitting improves embedding quality.

### Data Setup

The pretrained GloVe file is not included in this repository.

Place the following file at:

text models/glove.6B.300d.txt 

The current source is GloVe 6B, trained on Wikipedia 2014 and Gigaword 5, with 300-dimensional uncased vectors.

The current WordNet configuration is:

text include_synonyms=True include_hypernyms=False include_hyponyms=False 


---

## PART C. Evaluation

This module evaluates the quality of original and retrofitted word vectors.

### Files

```text
src/eval.py        # word similarity evaluation: WS-353, SimLex-999, RG-65
src/eval_sst.py    # sentiment analysis evaluation: SST-2 (auto-downloaded from HuggingFace)
datasets/combined.csv      # WS-353 (353 word pairs, human similarity scores)
datasets/SimLex-999.txt    # SimLex-999 (999 word pairs, semantic similarity)
datasets/rg65.txt          # RG-65 (65 word pairs, Rubenstein & Goodenough 1965)
```

### Evaluation Datasets

| Dataset | Pairs | Score range | Focus |
|---------|-------|-------------|-------|
| WS-353 | 353 | 0–10 | General word relatedness |
| SimLex-999 | 999 | 0–10 | Semantic similarity only |
| RG-65 | 65 | 0–4 | Classic semantic similarity benchmark |
| SST-2 | 67,349 train / 872 val | binary | Sentiment classification |

### Word Similarity Evaluation

For each dataset, we compute cosine similarity between word pairs using both original and retrofitted vectors, then measure Spearman rho correlation with human-annotated scores.

Run:

```bash
python src/eval.py
```

Results (GloVe 6B 300d, 50,000 words, 10 iterations, WordNet synonyms):

| Dataset | Original GloVe | Retrofitted | Δ |
|---------|---------------|-------------|---|
| WS-353 | 0.631 | 0.645 | +0.014 |
| SimLex-999 | 0.372 | 0.443 | +0.071 |
| RG-65 | 0.793 | 0.820 | +0.027 |

### Sentiment Analysis Evaluation

Sentences are represented as the average of their word vectors. A logistic regression classifier is trained on SST-2 and evaluated on the validation set.

Run:

```bash
python src/eval_sst.py
```

Results:

| | Accuracy |
|--|----------|
| Original GloVe | 76.61% |
| Retrofitted | 78.21% |
| Improvement | +1.60% |

### Coverage

We load the first 50,000 words from GloVe. Coverage on evaluation datasets:

- WS-353: 348/353 (98.6%)
- SimLex-999: 995/999 (99.6%)
- RG-65: 61/65 (93.8%)

Missing pairs are skipped; results remain reliable.
