# Retrofitting Word Vectors

This project reproduces the retrofitting method from Faruqui et al. (2015).

## PART A. Data Loading and Preprocessing

This module prepares the data for retrofitting.

The first version uses（即论文第一版，对应`WN_syn = synonyms only`）:

```text
GloVe 6B 300d + English WordNet synonyms
```

This corresponds to the paper's `WN_syn` setting.

It prepares:

```python
embeddings: dict[str, np.ndarray]
graph: dict[str, set[str]]
```

之后我们会加以下组合（即论文的第二版，对应`WN_all = synonyms + hypernyms + hyponyms`）

Version 2:
```python
GloVe 6B 300d + English WordNet synonyms + hypernyms + hyponyms
```

Version 3:
```python
GloVe 6B 300d + PPDB
```

Or Version 4:
```python
GloVe 6B 300d + WordNet + PPDB
```
### Files

```text
src/utils.py              # load embeddings, save embeddings, cosine similarity
src/preprocessing.py      # build WordNet graph, filter OOV words

prepare_wn_syn.py         # checks full GloVe 300d + WordNet synonyms, 之后哲兮可以融到main.py，然后删掉
```

### Data Source

Download GloVe from the [Stanford GloVe project page](https://nlp.stanford.edu/projects/glove/).

Use the pretrained vectors named:

**Wikipedia 2014 + Gigaword 5**  
`glove.6B.zip`

Extract this file:

```text
glove.6B.300d.txt
```

Place it at （需要你们手动下载，然后放在models文件夹里:）

```text
models/glove.6B.300d.txt
```

The GloVe file is not uploaded to GitHub because it is too large.

### OOV Handling

Some words appear in WordNet but do not have a GloVe vector. Since retrofitting needs an initial vector for each updated word, these words cannot be directly updated.

In this first version, we use a simple filtering strategy:

- keep only words that appear in the GloVe vocabulary;
- remove WordNet edges whose neighbor does not have a GloVe vector;
- remove graph nodes that have no remaining neighbors after filtering.

This gives a usable WordNet synonym graph aligned with the GloVe vocabulary.

### Check The Data Pipeline

Run a smaller check with the first 50,000 GloVe words:

```bash
python3 prepare_wn_syn.py --max-words 50000
```

Run the full version with:

```bash
python3 prepare_wn_syn.py
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

### Current Limitations

- The current experiments use prefixes of the GloVe file rather than random vocabulary samples.
- GloVe vectors are currently not L2-normalized before retrofitting.
- WordNet relations are aggregated across word senses, which may introduce polysemy-related noise.
- oov_after_filtering = 0 only describes the graph after vocabulary filtering.
- The current dictionary-based float64 representation has substantial memory overhead.
- The full 400,000-word vocabulary has not been tested.
- Original-versus-retrofitted evaluation has not yet been completed.
- Retrofitted vectors are not yet saved because the output format and vocabulary-order contract must first be agreed with the evaluation member.

### Next Steps

1. Agree on an output format and vocabulary-order contract.
2. Save and reload a small retrofitted output as a round-trip test.
3. Provide original and retrofitted embeddings to the evaluation pipeline.
4. Establish an original-GloVe baseline.
5. Compare controlled configurations such as iteration count, normalization policy, lexical relation types, and weighting strategies.
6. Conduct error analysis for noisy WordNet edges.

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