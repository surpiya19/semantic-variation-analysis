# IDS 570: Text as Data — Term Project
**Author:** Supriya Nannapaneni  
**Concept:** *trade*  
**Corpus:** Early Modern English Political Economy Texts (23 documents, c. 1580–1776)  

---

## Project Overview

This project investigates how the meaning of the word *trade* varies across a corpus of 23 early modern English political economy texts using four NLP methods: Named Entity Recognition (spaCy), contextual embeddings (BERT), supervised classification (logistic regression), and interpretive close reading.

The central finding is that *trade* does not have a single meaning in this corpus — it designates three structurally distinct frameworks that succeed one another historically:

- **Monetary** (1580–1622): trade as currency exchange and bullion flow (Malynes)
- **Colonial** (1660s): trade as Atlantic plantation and settlement activity
- **Institutional** (1680s–1776): trade as chartered corporate governance, culminating in Adam Smith's systematic political economy

---

## Repository Structure

```
ids570-term-project/
│
├── IDS570_Term_Project.ipynb     # Main notebook (all 5 steps + lit review)
│
├── texts/                        # Corpus .txt files (23 documents)
│   ├── A06785.txt
│   ├── A06786.txt
│   ├── ...
│   └── wealth.txt
│
├── figures/                      # All output figures
│   ├── 01_keyword_dist.png
│   ├── 02_gpe_cooccurrence.png
│   ├── 03_umap_clusters.png
│   ├── 04_confusion_matrix.png
│   ├── 04_top_features.png
│   ├── 04_predicted_by_group.png
│   └── 05_meaning_by_decade.png
│
├── trade_bert_embeddings.npy     # Saved BERT embeddings (2432 x 768)
│
└── README.md
```

---

## Setup

### Requirements

```bash
pip install spacy transformers torch umap-learn scikit-learn matplotlib seaborn pandas numpy tqdm
python -m spacy download en_core_web_sm
```

Python 3.10+ recommended.

### Running the notebook

1. Clone this repo
2. Place your corpus `.txt` files in the `texts/` folder
3. Set `CORPUS_DIR = './texts'` in Step 1 of the notebook (already set)
4. Run cells in order

> **Note:** The BERT embedding cell (Step 3) takes ~7 minutes on CPU for 2,432 contexts. Embeddings are saved to `trade_bert_embeddings.npy` after the first run — subsequent runs can load from disk and skip recomputation.

---

## Methods Summary

| Step | Method | Tool | Output |
|------|--------|------|--------|
| 0 | Literature review | — | 1-page review, 5 sources |
| 1 | Keyword extraction | pandas, regex | 2,432 ±2-sentence context windows |
| 2 | Named Entity Recognition | spaCy `en_core_web_sm` | GPE co-occurrence table + figure |
| 3 | Contextual embeddings | BERT `bert-base-uncased` + UMAP | k=4 clusters, 2D projection |
| 4 | Supervised classification | TF-IDF + Logistic Regression | macro F1 = 0.829 |
| 5 | Synthesis | Close reading + all above | Central claim + timeline figure |

---

## Key Results

- **2,432** total occurrences of *trade* across 23 documents
- **Classifier performance:** macro F1 = 0.829 (monetary F1 = 0.953, institutional F1 = 0.855, colonial F1 = 0.680)
- **Timeline finding:** monetary meaning dominates 1580–1620; institutional meaning rises in the 1680s and accounts for 57.6% of Adam Smith's *trade* contexts by 1776
- **Hardest cases:** Misselden's *Circle of Commerce* (1623) — the hinge text where *trade* shifts from a monetary-valuation problem to an institutional-governance problem

---

## Normalization Decisions

Consistent with the data exploration assignment:
- Long-S replacement (`ſ` → `s`)
- Removal of non-Latin placeholder tags (`<in non-Latin alphabet>`)
- Removal of typographic artefact characters (`● ▪ ◊`)
- Variant spellings (`mony`/`money`, `vpon`/`upon`) deliberately preserved

---

## Notes

- `trade_bert_embeddings.npy` is included so you don't need to rerun the BERT cell
- Weak labels for classification are generated programmatically — see Step 4 for keyword lists and rationale
- The `DOC_GROUPS` and `DOC_DATES` dictionaries in the notebook map document IDs to groups and approximate dates based on the data exploration report

---

## References

- Devlin et al. (2019). BERT. *NAACL-HLT 2019.*
- Ehrmann et al. (2021). NER in historical documents. *ACM Computing Surveys*, 54(8).
- Grimmer, Roberts & Stewart (2022). *Text as Data.* Princeton University Press.
- Hamilton, Leskovec & Jurafsky (2016). Diachronic word embeddings. *ACL 2016.*
- Poovey (1998). *A History of the Modern Fact.* University of Chicago Press.
