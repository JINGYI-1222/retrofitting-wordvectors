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
