# Text Emotion Classification: Transformers vs Recurrent Networks

Notebook experiments compare Transformer, BiGRU, BiGRU with attention and BiLSTM using BPE and Unigram tokenisation on a tweet-emotion dataset. **Historical results have not been rerun after the corrections.**

## Corrections (November 2026)

- Transformer sweep's training call was unreachable after a continue.
- Transformer pooling now excludes padded positions. **Previously saved Transformer checkpoints still use the old architecture and must be retrained.**
- Both notebooks now use the same deterministic cleaned 80/10/10 split. Duplicate texts and examples with conflicting labels are excluded **before** splitting. The old recorded scores are not directly comparable to new metrics.
- Displayed emotion label "sadnes" is corrected to "sadness".
- BiGRU inference supplies one input and UMAP code no longer references a commented-out third model.
- Checked-in SentencePiece tokenisers belong to historical experiments. The corrected notebook now trains separate 16,000-piece v2 tokenisers from the training partition only and fingerprints the exact corpus, refusing to reuse a vocabulary from a different split.
- scripts/top10.py defaults to a read-only dry run, including top-N directory cleanup.

## Setup and testing

Python 3.11 recommended. From the repository root:

    python -m venv .venv
    # Activate the environment on your OS
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python baseline_tfidf.py --data data.parquet
    jupyter lab

The tracked dataset is data.parquet. The notebook links to a Kaggle example; provenance, licence and the precise data revision should be verified and recorded before re-use.

## Interpretation

Saved notebook outputs are historical. New preprocessing and masked pooling change what is measured; do not reuse prior validation macro-F1 as corrected results. The standalone baseline_tfidf.py uses the same cleaned partition for TF-IDF + Logistic Regression; it must be run to obtain its scores. Per-class metrics and independent seeds remain important for comparisons. Exact text de-duplication does not remove near-duplicates or potential pretraining overlap.

The inference notebook reads v2 tokenisers and v2 best-checkpoint paths. It intentionally refuses to load historical checkpoints without the corrected retraining runs. The old files remain in Git history for archival inspection; avoid mixing old saved metrics with the new evaluation.
