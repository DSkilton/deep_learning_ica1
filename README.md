# Tweet emotion classification: Transformers versus recurrent models

This exploratory research compares Transformer, BiGRU, attention-augmented
BiGRU and BiLSTM classifiers, with BPE and Unigram tokenisation.

## Correctness and reproducibility changes (5 November 2026)

The historical notebook scores, models and tokenisers **must not be treated as
reproduced results** from the corrected code. The old pipeline contained an
unreachable Transformer training call, pooling over padded positions, a potential
overlap of duplicate texts across partitions, and inference errors.

- `data_splits.make_splits` removes duplicate texts **before** an 80/10/10
  stratified split; case/whitespace duplicates are detected, and texts with
  inconsistent labels are excluded with a warning.
- Split logic is shared between `main.ipynb` and `use_model.ipynb`.
- New `m_*_group_split_v2` tokenisers are fitted **only** to training text.
  Their SHA-256 training-corpus manifests prevent inadvertent reuse against
  different training data. Historical 16k-vocabulary models are **not**
  silently substituted for the new requested 8k models.
- The Transformer uses its token mask during attention **and** final mean pooling.
- Corrected training runs use `logs/fit/group_split_v2/`, keeping them
  separate from saved historical checkpoints. The full parameter sweeps are
  disabled by default: set `RUN_SWEEPS = True` explicitly to execute them.
- `use_model.ipynb` loads the best new checkpoints selected on validation
  macro-F1, and passes one input to the BiGRU rather than two.
- `scripts/top10.py` is a dry run by default. `--apply` copies results,
  while deleting experiment directories additionally requires
  `--apply --delete-low-scores`. No existing top-run copies are cleared.

## Getting started

Use Python 3.11 in a dedicated environment (Linux/WSL2 is recommended
for NVIDIA GPU TensorFlow). The original `requirements.txt` is a legacy
environment export containing an incompatible Keras/TensorFlow pairing.
The following is a simpler compatible **target**, not a fully hardware-tested
lockfile:

```bash
python -m venv .venv
python -m pip install -r requirements-training.txt
python -m pip install -r requirements-checks.txt
python -m pytest -q
```

Then open `main.ipynb` and run cells from a clean kernel. Set
`RUN_SWEEPS = True` only when ready for the considerable training time and
disk usage. Once new training runs exist, `use_model.ipynb` can load their
best validation checkpoints and produce two-model UMAP visualisations.

The existing `data.parquet` and historical training artefacts are retained
for provenance. New model outputs and regenerated tokenisers are ignored by
Git; archive their manifests and data hashes with any published experiment.

## Research limitations

These are benchmark experiments on labelled tweets, **not clinical or
high-stakes emotion judgements**. Neutral or mixed emotions may be absent.
Exact duplicate handling does not detect semantic paraphrases or all possible
dataset contamination. Perform per-class error analysis, cross-seed repeats,
and a TF-IDF/Logistic Regression baseline, and report a genuinely untouched
test partition before claiming a winning architecture. Earlier recorded
macro-F1 values are for historical reference only and cannot be compared
directly with newly split results without a fresh run.
