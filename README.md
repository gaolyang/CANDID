# CANDID

CANDID is a continual adaptive network defense framework for intrusion detection
in dynamic open environments. This repository is a curated, reproducible release
of the code used for the paper **"CANDID: A Continual Adaptive Network Defense
Framework for Intrusion Detection in Dynamic Open Environments"**.

The experimental configuration in the scripts matches the hyperparameters reported
in the paper (Table 3 / Table 13 / Table 14), so running these scripts reproduces
the reported results (over 5 independent random seeds).

## What Is Included

- Dual-consistency benign drift filtering: selects reliable drift-benign samples
  using feature-space deviation and decision-space uncertainty.
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
- `candid_lib/dataset_specification.py`: unified flow-format feature definitions.

## Experimental Configuration (paper Table 3 / 13 / 14)

| Hyperparameter | Value |
|---|---|
| Epochs | 250 |
| Batch size | 256 |
| Optimizer | Adam, learning rate 1e-4 |
| Sliding window | 8 flow records |
| SupCon (τsc, λsc) | (0.07, 0.2) |
| Distillation weight λkd | 10.0 |
| Decision thresholds (δl, δu) | (0.3, 0.7) |
| Replay memory budget K | 200 |
| IQR coefficient | 1.5 |
| Drift-ratio trigger ρmax | 0.30 |
| Tightening quantile qtight | 0.70 |
| Drift-benign range [Nmin, Nmax] | [2000, 30000] |
| Random exploration rexp | 0.10 |
| Random seeds | 42, 52, 62, 72, 82 |

Task construction (paper Table 2):

- **CICIDS2018**: base task {DoS, DDoS, Infiltration}; incremental tasks
  {Web Attacks, BruteForce, Bot}; benign traffic split into 4 chronological groups.
- **UNSW-NB15**: base task {Fuzzers, Analysis, Backdoor, DoS, Reconnaissance,
  Shellcode}; incremental tasks {Worms, Exploits, Generic}; benign traffic split
  into 4 chronological groups.

## Run

Install dependencies (Python 3.8, TensorFlow 2.15):

```bash
pip install -r requirements.txt
```

Edit the dataset path (or pass it via `--dataset`), then run from this folder so
`candid_lib` is importable:

```bash
# CICIDS2018 (5 seeds, aggregates mean +/- std)
python scripts/run_cicids2018_candid.py \
    --dataset /path/to/NF-CICIDS2018-v3.csv \
    --seeds 42 52 62 72 82 \
    --summary-csv results_cicids2018.csv

# UNSW-NB15
python scripts/run_unsw_nb15_candid.py \
    --dataset /path/to/NF-UNSW-NB15-v3.csv \
    --seeds 42 52 62 72 82 \
    --summary-csv results_unsw_nb15.csv
```

For a quick smoke test (not the paper configuration), override the epoch count:

```bash
python scripts/run_cicids2018_candid.py --dataset ... --epochs 1 --seeds 42
```

## Method Summary

1. Build a continual task stream from temporally grouped benign traffic and
   staged attack categories.
2. Train the base task with benign traffic and initial attack categories.
3. For later tasks, filter current benign samples with dual consistency:
   feature drift against the benign profile (`d > T` via the IQR boundary) plus
   decision uncertainty (`δl < p < δu`).
4. Update the model with current attacks, selected drift-benign samples, and
   representative replay memory, optimizing BCE + λsc·SupCon + λkd·selective KD.
5. Refresh the benign profile and replay memory after each task.

## Licensing

The low-level Transformer framework files (`candid_lib/*`) are derived from the
original FlowTransformer codebase, which is distributed under the **AGPL-3.0**
license. The original license text is kept in
`LICENSE-FlowTransformer-AGPL-3.0.txt`. The CANDID-specific code in this repository
is made available under the same **AGPL-3.0** license to remain compatible with the
derived FlowTransformer components.
