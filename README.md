# CANDID

CANDID is a continual intrusion detection framework for dynamic traffic
environments. The implementation here is a curated release of the code used for
the paper description in `cas-dc-template.pdf`, with comparison methods,
ablation-only switches, plotting scripts, notebooks, cached models, and
experiment artifacts removed.

## What Is Included

- Dual-consistency benign drift filtering: selects reliable drift-benign samples
  using feature-space deviation and model decision uncertainty.
- Transformer-based flow encoder: preprocesses tabular flow records, encodes
  record-level features, and applies a lightweight Transformer stack.
- Protected incremental updating: combines binary detection loss, supervised
  contrastive loss, representative replay, and selective knowledge distillation.
- Dataset entry scripts for CICIDS2018 and UNSW-NB15 in the unified flow format.

## Code Layout

- `scripts/run_cicids2018_candid.py`: CICIDS2018 continual task construction,
  CANDID training loop, zero-day evaluation, and task-aware evaluation.
- `scripts/run_unsw_nb15_candid.py`: UNSW-NB15 continual task construction and
  CANDID training/evaluation loop.
- `candid_lib/engine_cicids2018.py`: CANDID FlowTransformer engine variant used
  by the CICIDS2018 script.
- `candid_lib/engine_unsw_nb15.py`: CANDID FlowTransformer engine variant used
  by the UNSW-NB15 script.
- `candid_lib/preprocessing.py`: numerical and categorical flow preprocessing.
- `candid_lib/input_encodings.py`: record-level feature embedding.
- `candid_lib/basic_transformers.py` and `candid_lib/transformer_blocks/`:
  Transformer encoder/decoder components.
- `candid_lib/classification_heads.py`: classification heads, including the
  last-token head used by the scripts.
- `candid_lib/dataset_specification.py`: unified flow-format feature
  definitions.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then edit the dataset path near the bottom of the script you want to run:

```python
dataset_path = r"/home/test/NF-CICIDS2018-v3.csv"
dataset_path = r"/home/test/NF-UNSW-NB15-v3.csv"
```

Run from this folder so `candid_lib` is importable:

```bash
python scripts/run_cicids2018_candid.py
python scripts/run_unsw_nb15_candid.py
```

## Method Summary

The main CANDID flow is:

1. Build a continual task stream from temporally grouped benign traffic and
   staged attack categories.
2. Train the base task with benign traffic and initial attack categories.
3. For later tasks, filter current benign samples with dual consistency:
   feature drift against the benign profile plus model uncertainty.
4. Update the model with current attacks, selected drift-benign samples, and
   representative replay memory.
5. Apply selective distillation on historical attack classes and supervised
   contrastive learning to improve representation separation.
6. Refresh the benign profile and replay memory after each task.

## Notes

This directory intentionally does not include comparison-method implementations
or experiment plotting scripts. The low-level Transformer framework files are
derived from the original FlowTransformer codebase; the original AGPL-3.0
license text is kept in `LICENSE-FlowTransformer-AGPL-3.0.txt`.
