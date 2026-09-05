# CANDID - Continual Intrusion Detection on CICIDS2018
#
# Reproduces the CANDID experimental configuration described in
# "CANDID: A Continual Adaptive Network Defense Framework for Intrusion
# Detection in Dynamic Open Environments" (Table 3 / Table 13 / Table 14).
#
# Task stream (Table 2, CICIDS2018):
#   T0 Base : {DoS, DDoS, Infiltration}             + benign group B0
#   T1 Incr.: {Web Attacks}                         + benign group B1
#   T2 Incr.: {BruteForce}                          + benign group B2
#   T3 Incr.: {Bot}                                 + benign group B3
#
# Training configuration (paper Table 3 / 13 / 14):
#   epochs=250, batch_size=256, Adam lr=1e-4, window_size=8
#   SupCon (tau_sc=0.07, lambda_sc=0.2), distillation lambda_kd=10.0
#   decision thresholds (delta_l, delta_u) = (0.3, 0.7)
#   replay memory budget K=200, IQR=1.5, rho_max=0.30, q_tight=0.70
#   [Nmin,Nmax]=[2000,30000], r_exp=0.10
#
# The experiment is repeated over independent random seeds
# (default: 42, 52, 62, 72, 82) and results are aggregated as mean +/- std.
#
# Run (from the candid/ root so candid_lib is importable):
#   python scripts/run_cicids2018_candid.py --dataset /path/to/NF-CICIDS2018-v3.csv
#   python scripts/run_cicids2018_candid.py --dataset ... --seeds 42 52 62 72 82

import argparse
import copy
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf

from candid_lib.engine_cicids2018 import FlowTransformer
from candid_lib.dataset_specification import NamedDatasetSpecifications
from candid_lib.flow_transformer_parameters import FlowTransformerParameters
from candid_lib.model_input_specification import ModelInputSpecification
from candid_lib.preprocessing import StandardPreProcessing
from candid_lib.basic_transformers import BasicTransformer
from candid_lib.classification_heads import LastTokenClassificationHead
from candid_lib.input_encodings import RecordLevelEmbed


# ============================================================================
# Command line arguments
# ============================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="CANDID continuous intrusion detection on CICIDS2018."
    )
    parser.add_argument(
        "--dataset",
        default="/home/test/NF-CICIDS2018-v3.csv",
        help="Path to the unified-flow CICIDS2018 dataset (NF-CICIDS2018-v3.csv).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62, 72, 82],
        help="Independent random training seeds (default: 42 52 62 72 82).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of incremental-training epochs per task (paper: 250).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (paper: 256).",
    )
    parser.add_argument(
        "--gpu",
        default="1",
        help="CUDA_VISIBLE_DEVICES value.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Append per-task results to this CSV file.",
    )
    return parser


# ============================================================================
# Hyperparameters (paper Table 3 / Table 13)
# ============================================================================
EPOCHS = 250
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
SLIDING_WINDOW = 8

SUPCON_LAMBDA = 0.2      # lambda_sc (paper)
SUPCON_TEMP = 0.07       # tau_sc (paper)
DISTILL_LAMBDA = 10.0    # lambda_kd (paper)

# Dual-consistency decision uncertainty bounds (delta_l, delta_u) = (0.3, 0.7)
CONFIDENCE_THRESHOLD = 0.3       # delta_l
DRIFT_ATTACK_PROB_UPPER = 0.7    # delta_u

# Replay memory
MEMORY_PER_CLASS = 200   # K (paper)
BENIGN_MEMORY_SIZE = 2000

# Evaluation split
ORIGINAL_EVAL_PERCENT = 0.2
EVAL_SPLIT_MODE = "stratified"

# Fixed decision threshold for binary evaluation (no test-set calibration)
FIXED_DECISION_THRESHOLD = 0.5

# Feature extraction
FEATURE_EXTRACTION_BATCH_SIZE = 1024
MAX_MEMORY_CANDIDATES_PER_ATTACK = 500
MAX_MEMORY_CANDIDATES_BENIGN = 2000
MAX_DRIFT_FILTER_CANDIDATES = 50000

# CANDID method switches (full CANDID configuration)
METHOD_NAME = "CANDID"
USE_REPLAY = True
USE_DISTILL = True
USE_SUPCON = True
USE_DRIFT_FILTER = True
UPDATE_MEMORY = True
UPDATE_BENIGN_PROFILE = True


# ============================================================================
# Attack-label grouping (CICIDS2018)
# ============================================================================
def group_attack_labels(label):
    s_label = str(label).strip().lower()

    if "benign" in s_label:
        return "Benign"
    if "ddos" in s_label:
        return "DDoS"
    if "dos" in s_label:
        return "DoS"
    if (
        "web" in s_label
        or "sql" in s_label
        or "xss" in s_label
        or "injection" in s_label
    ):
        return "Web Attacks"
    if "brute" in s_label or "ssh" in s_label or "ftp" in s_label:
        return "BruteForce"
    if "infilteration" in s_label:
        return "Infiltration"
    if "bot" in s_label:
        return "Bot"
    return "Other"


# ============================================================================
# Data preparation
# ============================================================================
def prepare_data_custom(dataset_path, dataset_spec):
    print("=" * 20 + " Data Preparation (Attack Stratified + Benign Time Grouping) " + "=" * 20)
    df_full = pd.read_csv(dataset_path)

    if "attack_cat" in df_full.columns:
        dataset_spec.class_column = "attack_cat"

    time_columns = [
        c for c in ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
        if c in df_full.columns
    ]
    if time_columns:
        for col_name in time_columns:
            df_full[col_name] = pd.to_numeric(df_full[col_name], errors="coerce")
        print(f"Detected benign grouping time columns: {time_columns}")

    df_full["Grouped_Label"] = df_full[dataset_spec.class_column].apply(group_attack_labels)
    spec_grouped = copy.deepcopy(dataset_spec)
    spec_grouped.class_column = "Grouped_Label"
    split_mode = str(EVAL_SPLIT_MODE).strip().lower()

    grouped_labels = df_full["Grouped_Label"].unique()
    available_labels = set(grouped_labels)

    # Task construction (paper Table 2)
    base_attacks = [
        a for a in ["DoS", "DDoS", "Infiltration"] if a in available_labels
    ]
    inc_attacks = [
        a for a in ["Web Attacks", "BruteForce", "Bot"] if a in available_labels
    ]
    total_tasks = 1 + len(inc_attacks)

    def split_dataframe_for_eval(df_subset, mode, random_state, test_ratio, sort_columns=None):
        df_subset = df_subset.copy().reset_index(drop=True)
        n_total = len(df_subset)
        if n_total == 0:
            return df_subset.copy(), df_subset.copy()

        n_test = int(np.ceil(n_total * test_ratio))
        if n_total > 1:
            n_test = max(1, min(n_total - 1, n_test))
        else:
            n_test = 1

        if mode == "last":
            if sort_columns:
                df_subset = df_subset.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)
            split_at = n_total - n_test
            return (
                df_subset.iloc[:split_at].copy().reset_index(drop=True),
                df_subset.iloc[split_at:].copy().reset_index(drop=True),
            )

        if mode in {"random", "stratified"}:
            shuffled = df_subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            split_at = n_total - n_test
            return (
                shuffled.iloc[:split_at].copy().reset_index(drop=True),
                shuffled.iloc[split_at:].copy().reset_index(drop=True),
            )

        raise ValueError(f"Unsupported split mode: {mode}")

    if split_mode == "stratified":
        print(
            f"Evaluation split mode: stratified per attack class + chronological benign "
            f"groups (test_ratio={ORIGINAL_EVAL_PERCENT:.0%})"
        )
        pool_train = {}
        pool_test = {}

        benign_all = df_full[df_full["Grouped_Label"] == "Benign"].copy().reset_index(drop=True)
        if benign_all.empty:
            raise ValueError("No Benign samples found in the dataset.")
        if time_columns:
            benign_all = benign_all.sort_values(by=time_columns, kind="stable").reset_index(drop=True)
        else:
            print("Warning: benign chronological grouping falls back to current file order.")

        benign_chunks_full = [
            chunk.reset_index(drop=True)
            for chunk in np.array_split(benign_all, total_tasks)
            if len(chunk) > 0
        ]
        if len(benign_chunks_full) < total_tasks:
            raise ValueError(
                f"Benign chronological groups are insufficient: expected {total_tasks}, "
                f"got {len(benign_chunks_full)}."
            )

        benign_train_groups = []
        benign_test_groups = []
        for idx, chunk in enumerate(benign_chunks_full):
            train_chunk, test_chunk = split_dataframe_for_eval(
                chunk,
                mode="last",
                random_state=42 + idx,
                test_ratio=ORIGINAL_EVAL_PERCENT,
                sort_columns=None,
            )
            benign_train_groups.append(train_chunk)
            benign_test_groups.append(test_chunk)
            print(f"[Benign Group {idx}] train={len(train_chunk)}, test={len(test_chunk)}")

        pool_train["Benign"] = pd.concat(benign_train_groups, ignore_index=True).reset_index(drop=True)
        pool_test["Benign"] = pd.concat(benign_test_groups, ignore_index=True).reset_index(drop=True)

        for cls_idx, cls in enumerate(grouped_labels):
            if cls == "Benign":
                continue
            df_cls = df_full[df_full["Grouped_Label"] == cls].copy().reset_index(drop=True)
            train_cls, test_cls = split_dataframe_for_eval(
                df_cls,
                mode="stratified",
                random_state=42 + 100 + cls_idx,
                test_ratio=ORIGINAL_EVAL_PERCENT,
            )
            pool_train[cls] = train_cls
            pool_test[cls] = test_cls
            print(f"[Attack Split] {cls}: train={len(train_cls)}, test={len(test_cls)}")

        df_train = pd.concat(pool_train.values(), ignore_index=True).reset_index(drop=True)
        df_test = pd.concat(pool_test.values(), ignore_index=True).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported EVAL_SPLIT_MODE: {EVAL_SPLIT_MODE}. Use 'stratified'.")

    base_attacks = [a for a in base_attacks if a in pool_train and not pool_train[a].empty]
    inc_attacks = [a for a in inc_attacks if a in pool_train and not pool_train[a].empty]
    total_tasks = 1 + len(inc_attacks)

    if "Benign" not in pool_train or pool_train["Benign"].empty:
        raise ValueError("No Benign samples found in the training split.")

    benign_train_groups = [chunk.reset_index(drop=True) for chunk in benign_train_groups[:total_tasks]]
    benign_test_groups = [chunk.reset_index(drop=True) for chunk in benign_test_groups[:total_tasks]]

    if len(benign_train_groups) < total_tasks or len(benign_test_groups) < total_tasks:
        raise ValueError(
            f"Benign task groups are insufficient after split: train_groups="
            f"{len(benign_train_groups)}, test_groups={len(benign_test_groups)}, "
            f"expected={total_tasks}."
        )

    tasks = []

    def build_task(attack_names, task_name, benign_group_idx):
        att_dfs = [pool_train[a] for a in attack_names if a in pool_train]
        if not att_dfs:
            return None

        df_att = pd.concat(att_dfs, ignore_index=True)
        df_benign_chunk = benign_train_groups[benign_group_idx].copy().reset_index(drop=True)
        df_benign_test_chunk = benign_test_groups[benign_group_idx].copy().reset_index(drop=True)
        task_data = pd.concat([df_att, df_benign_chunk], ignore_index=True)

        return {
            "name": task_name,
            "data": task_data,
            "spec": spec_grouped,
            "new_labels": attack_names,
            "benign_group_idx": benign_group_idx,
            "task_benign_test": df_benign_test_chunk,
            "test_pool": pool_test,
            "full_test": df_test.copy(),
        }

    next_benign_group_idx = 0
    if base_attacks:
        print(f"Building Task 0 (Benign Group {next_benign_group_idx})...")
        t0 = build_task(base_attacks, "Task_0_Base", benign_group_idx=next_benign_group_idx)
        if t0:
            tasks.append(t0)
            next_benign_group_idx += 1
    else:
        print("Warning: no base attacks available in the training split; Task 0 skipped.")

    for att in inc_attacks:
        task_idx = len(tasks)
        benign_group_idx = next_benign_group_idx
        print(f"Building Incremental Task for {att} (Benign Group {benign_group_idx})...")
        ti = build_task([att], f"Task_{task_idx}_{att}", benign_group_idx=benign_group_idx)
        if ti:
            tasks.append(ti)
        next_benign_group_idx += 1

    for i, task in enumerate(tasks):
        vc = task["data"][spec_grouped.class_column].value_counts(dropna=False).to_dict()
        print(f"[Audit] Task {i} ({task['name']}): labels={ {str(k): int(v) for k, v in vc.items()} }")

    return tasks, StandardPreProcessing(n_categorical_levels=32)


# ============================================================================
# Class manager and rehearsal memory
# ============================================================================
class ClassManager:
    def __init__(self):
        self.class_to_idx = {"Benign": 0}
        self.idx_to_class = {0: "Benign"}
        self.known_classes = 1

    def add_classes(self, labels):
        for label in labels:
            if label not in self.class_to_idx and label != "Benign":
                idx = self.known_classes
                self.class_to_idx[label] = idx
                self.idx_to_class[idx] = label
                self.known_classes += 1


class RehearsalMemory:
    def __init__(self, memory_per_class, benign_max):
        self.memory_per_class = memory_per_class
        self.benign_memory_size = benign_max
        self.data = pd.DataFrame()
        self.labels = pd.Series(dtype=int)

    def _downsample_candidates(self, df_candidates, max_candidates):
        if len(df_candidates) <= max_candidates:
            return df_candidates
        return df_candidates.sample(n=max_candidates, random_state=42).reset_index(drop=True)

    def _select_representative(self, df_candidates, max_size, feature_extractor):
        """Spherical-centroid representative sample selection (Algorithm 2)."""
        if len(df_candidates) <= max_size:
            return df_candidates

        max_candidates = max(
            max_size,
            MAX_MEMORY_CANDIDATES_BENIGN
            if max_size == self.benign_memory_size
            else MAX_MEMORY_CANDIDATES_PER_ATTACK,
        )
        df_candidates = self._downsample_candidates(df_candidates, max_candidates)

        feats = feature_extractor(df_candidates)
        if len(feats) == 0:
            return df_candidates.sample(n=max_size, random_state=42)

        centroid = np.mean(feats, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

        sims = np.dot(feats, centroid)
        distances = 1.0 - sims

        closest_indices = np.argpartition(distances, max_size - 1)[:max_size]
        closest_indices = closest_indices[np.argsort(distances[closest_indices])]

        return df_candidates.iloc[closest_indices].reset_index(drop=True)

    def add(self, new_data, new_labels_idx, class_manager, feature_extractor):
        num_known_classes = class_manager.known_classes
        if num_known_classes == 0:
            return

        final_data_list = []
        final_labels_list = []

        # 1. Benign (label 0)
        current_benign_mask = (new_labels_idx == 0)
        old_benign_mask = (self.labels == 0)

        candidates_benign = pd.concat(
            [
                self.data.loc[old_benign_mask] if not self.data.empty else pd.DataFrame(),
                new_data.loc[current_benign_mask],
            ],
            ignore_index=True,
        )
        if not candidates_benign.empty:
            kept = self._select_representative(candidates_benign, self.benign_memory_size, feature_extractor)
            final_data_list.append(kept)
            final_labels_list.append(pd.Series([0] * len(kept)))

        # 2. Attacks (label > 0)
        for class_id in range(1, num_known_classes):
            new_mask = (new_labels_idx == class_id)
            old_mask = (self.labels == class_id)

            candidates_attack = pd.concat(
                [
                    self.data.loc[old_mask] if not self.data.empty else pd.DataFrame(),
                    new_data.loc[new_mask],
                ],
                ignore_index=True,
            )
            if not candidates_attack.empty:
                kept = self._select_representative(candidates_attack, self.memory_per_class, feature_extractor)
                final_data_list.append(kept)
                final_labels_list.append(pd.Series([class_id] * len(kept)))

        if final_data_list:
            self.data = pd.concat(final_data_list, ignore_index=True)
            self.labels = pd.concat(final_labels_list, ignore_index=True)

    def get(self):
        return self.data, self.labels


# ============================================================================
# Incremental learner
# ============================================================================
class IncrementalLearner:
    def __init__(self, model_builder_fn, pre_processor, ft_components):
        self.model_builder_fn = model_builder_fn
        self.pre_processor = pre_processor
        self.ft_components = ft_components
        self.class_manager = ClassManager()
        self.memory = RehearsalMemory(MEMORY_PER_CLASS, BENIGN_MEMORY_SIZE)
        self.model = None
        self.prev_model = None
        self.task_data_info = []
        self.global_input_spec = None
        self.benign_center = None
        self.benign_threshold = 0.0
        self.decision_threshold = FIXED_DECISION_THRESHOLD

    def _extract_features(self, df_data, spec):
        if df_data.empty or self.model is None:
            return np.array([])

        temp_handler = FlowTransformer(self.pre_processor, **self.ft_components)
        temp_handler.model_input_spec = self.global_input_spec
        df, _ = temp_handler._load_preprocessed_dataset(
            "feature_extraction", df_data, spec, fit_preprocessor=False, evaluation_percent=0.0
        )
        if "__training" in df.columns:
            del df["__training"]
        if "__y" in df.columns:
            del df["__y"]

        X_np = df.values.astype(np.float32)
        X_np_padded = np.vstack([X_np, np.zeros((1, X_np.shape[1]), dtype=np.float32)])
        padding_idx = X_np.shape[0]
        features_list = []

        W = self.ft_components["params"].window_size
        window_offsets = np.arange(-W + 1, 1)

        for i in range(0, len(df), FEATURE_EXTRACTION_BATCH_SIZE):
            end = min(i + FEATURE_EXTRACTION_BATCH_SIZE, len(df))
            indices = np.arange(i, end)
            all_indices = indices[:, None] + window_offsets
            all_indices = np.clip(all_indices, 0, padding_idx - 1)
            windows_flat = X_np_padded[all_indices.ravel()]
            windows_np = windows_flat.reshape(len(indices), W, X_np_padded.shape[1])
            batch_dfs = [pd.DataFrame(win, columns=df.columns) for win in windows_np]
            batch_x = temp_handler.samplewise_to_featurewise(batch_dfs)

            _, feats = self.model.predict(batch_x, verbose=0)
            features_list.append(feats)

        del temp_handler, df, X_np, X_np_padded
        gc.collect()
        return np.vstack(features_list) if features_list else np.array([])

    def update_benign_profile(self, benign_data, spec, is_first_task=False):
        """Refresh benign center and IQR-based drift boundary (Stage 3)."""
        print("\n[Drift Updating] Recalculating Benign Profile with latest model...")
        feats = self._extract_features(benign_data, spec)
        if len(feats) == 0:
            return

        new_center = np.mean(feats, axis=0)
        self.benign_center = new_center / (np.linalg.norm(new_center) + 1e-9)

        sims = np.dot(feats, self.benign_center)
        dists = 1.0 - sims
        Q1, Q3 = np.percentile(dists, 25), np.percentile(dists, 75)
        new_threshold = Q3 + 1.5 * (Q3 - Q1)

        if is_first_task or self.benign_threshold == 0.0:
            self.benign_threshold = new_threshold
            print(f"[Drift Profile] Initialized -> Q1: {Q1:.4f} | Q3: {Q3:.4f}")
            print(f"[Drift Profile] Initial Threshold: {self.benign_threshold:.4f}")
        else:
            alpha = 0.7
            old_threshold = self.benign_threshold
            self.benign_threshold = alpha * self.benign_threshold + (1 - alpha) * new_threshold
            print(
                f"[Drift Profile] Updated Threshold (EMA) -> Old: {old_threshold:.4f} | "
                f"New: {self.benign_threshold:.4f}"
            )

    def filter_benign_via_drift(self, new_benign_data, spec):
        """Dual-consistency benign drift filter (Algorithm 1 / Eq. 1)."""
        if self.benign_center is None:
            return new_benign_data

        if len(new_benign_data) > MAX_DRIFT_FILTER_CANDIDATES:
            new_benign_data = new_benign_data.sample(
                n=MAX_DRIFT_FILTER_CANDIDATES, random_state=42
            ).reset_index(drop=True)

        print(f"\n[Drift Detection] Consistency Check on {len(new_benign_data)} samples...")

        model_to_use = self.prev_model if self.prev_model is not None else self.model
        temp_handler = FlowTransformer(self.pre_processor, **self.ft_components)
        temp_handler.model_input_spec = self.global_input_spec
        df, _ = temp_handler._load_preprocessed_dataset(
            "filtering", new_benign_data, spec, fit_preprocessor=False, evaluation_percent=0.0
        )
        if "__training" in df.columns:
            del df["__training"]
        if "__y" in df.columns:
            del df["__y"]

        X_np = df.values.astype(np.float32)
        X_np_padded = np.vstack([X_np, np.zeros((1, X_np.shape[1]), dtype=np.float32)])
        padding_idx = X_np.shape[0]
        features_list, probs_list = [], []

        W = self.ft_components["params"].window_size
        window_offsets = np.arange(-W + 1, 1)

        for i in range(0, len(df), FEATURE_EXTRACTION_BATCH_SIZE):
            end = min(i + FEATURE_EXTRACTION_BATCH_SIZE, len(df))
            indices = np.arange(i, end)
            all_indices = indices[:, None] + window_offsets
            all_indices = np.clip(all_indices, 0, padding_idx - 1)
            windows_flat = X_np_padded[all_indices.ravel()]
            windows_np = windows_flat.reshape(len(indices), W, X_np_padded.shape[1])
            batch_dfs = [pd.DataFrame(win, columns=df.columns) for win in windows_np]
            batch_x = temp_handler.samplewise_to_featurewise(batch_dfs)

            probs, feats = model_to_use.predict(batch_x, verbose=0)
            features_list.append(feats)
            probs_list.append(probs)

        if not features_list:
            return new_benign_data

        feats = np.vstack(features_list)
        probs = np.vstack(probs_list).reshape(-1)

        sims = np.dot(feats, self.benign_center)
        dists = 1.0 - sims

        # Feature-space drift (IQR boundary, with safety tightening)
        drift_ratio = np.mean(dists > self.benign_threshold)
        if drift_ratio > 0.3:  # rho_max
            adjusted_threshold = np.percentile(dists, 70)  # q_tight
            is_feature_drift = dists > adjusted_threshold
            print(
                f"  [Warning] Huge drift detected ({drift_ratio:.1%}). "
                f"Tightening boundary to {adjusted_threshold:.4f}"
            )
        else:
            is_feature_drift = dists > self.benign_threshold

        # Decision-space uncertainty band (delta_l, delta_u) = (0.3, 0.7)
        is_model_uncertain = (probs > CONFIDENCE_THRESHOLD) & (probs < DRIFT_ATTACK_PROB_UPPER)
        is_high_conf_attack = probs >= DRIFT_ATTACK_PROB_UPPER

        consistent_drift = is_feature_drift & is_model_uncertain
        simple_drift = is_feature_drift

        final_mask = consistent_drift
        random_mask = (np.random.rand(len(dists)) < 0.1) & (~is_high_conf_attack)  # r_exp
        final_mask = final_mask | random_mask

        # Cap the retained count (N_max)
        MAX_DRIFT_KEEP = 30000
        n_kept = np.sum(final_mask)
        if n_kept > MAX_DRIFT_KEEP:
            true_indices = np.where(final_mask)[0]
            keep_idx = np.random.choice(true_indices, size=MAX_DRIFT_KEEP, replace=False)
            final_mask = np.zeros_like(final_mask, dtype=bool)
            final_mask[keep_idx] = True
            print(
                f"  [Safety Clamp] Massive drift detected ({n_kept}). "
                f"Capped to {MAX_DRIFT_KEEP}."
            )

        # Guarantee a minimum retained count (N_min) from safe candidates
        MIN_SAMPLES = 2000
        n_kept = np.sum(final_mask)
        if n_kept < MIN_SAMPLES and len(new_benign_data) > MIN_SAMPLES:
            fallback_pool = np.where(simple_drift & ~final_mask & ~is_high_conf_attack)[0]
            needed = MIN_SAMPLES - n_kept
            if len(fallback_pool) > 0:
                add_idx = np.random.choice(
                    fallback_pool, size=min(len(fallback_pool), needed), replace=False
                )
                final_mask[add_idx] = True

        n_kept = np.sum(final_mask)
        if n_kept < MIN_SAMPLES and len(new_benign_data) > MIN_SAMPLES:
            discarded_indices = np.where(~final_mask & ~is_high_conf_attack)[0]
            needed = MIN_SAMPLES - n_kept
            if len(discarded_indices) >= needed:
                boost_indices = np.random.choice(discarded_indices, size=needed, replace=False)
                final_mask[boost_indices] = True

        filtered_data = new_benign_data.iloc[final_mask].reset_index(drop=True)

        print(f"[Drift Stats] Total: {len(new_benign_data)}")
        print(f"  -> Feature Drift: {np.sum(is_feature_drift)}")
        print(
            f"  -> Model Uncertain Band ({CONFIDENCE_THRESHOLD:.2f}, "
            f"{DRIFT_ATTACK_PROB_UPPER:.2f}): {np.sum(is_model_uncertain)}"
        )
        print(f"  -> Rejected High Attack-Prob: {np.sum(is_high_conf_attack)}")
        print(f"  -> Consistent Drift: {np.sum(consistent_drift)}")
        print(f"  -> Final Kept for Training: {len(filtered_data)}")

        del temp_handler, df, X_np, X_np_padded
        gc.collect()
        return filtered_data

    def evaluate_zero_day(self, task_idx, task_info):
        if task_idx == 0:
            return 0.0

        new_labels = task_info["new_labels"]
        print(
            f"\n>>> [Zero-Day Check] Testing model against NEW attacks: "
            f"{new_labels} (before training) <<<"
        )

        test_pool = task_info["test_pool"]
        dfs = []
        for label in new_labels:
            if label in test_pool and not test_pool[label].empty:
                dfs.append(test_pool[label])
        if not dfs:
            print("[Zero-Day Check] No test data available for new attacks.")
            return 0.0

        df_attack_only = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        spec = task_info["spec"]
        eval_class_to_idx = copy.deepcopy(self.class_manager.class_to_idx)
        next_idx = max(eval_class_to_idx.values(), default=-1) + 1
        for label in new_labels:
            if label not in eval_class_to_idx:
                eval_class_to_idx[label] = next_idx
                next_idx += 1

        handler = FlowTransformer(self.pre_processor, **self.ft_components)
        handler.m = self.model
        handler.model_input_spec = self.global_input_spec
        handler.load_dataset(
            f"zeroday_eval_{task_idx}",
            df_attack_only,
            spec,
            eval_class_to_idx,
            evaluation_percent=1.0,
            fit_preprocessor=False,
        )

        if handler.X is None or len(handler.X) == 0:
            return 0.0

        yielder = handler.get_batch_yielder(512, for_training=False, balanced=False)
        y_prob_list = []
        while True:
            bx, _ = yielder.get_batch()
            if not bx:
                break
            outputs = self.model.predict(bx, verbose=0)
            prob = outputs[0] if isinstance(outputs, list) else outputs
            y_prob_list.extend(np.array(prob).reshape(-1))

        y_prob = np.array(y_prob_list)
        preds = (y_prob > 0.5).astype(int)
        zero_day_recall = np.mean(preds)
        print(f"[Zero-Day Result] Intercepted {np.sum(preds)}/{len(preds)} unseen attacks.")
        print(f"[Zero-Day Result] ZeroDay_Recall: {zero_day_recall:.4f}")

        handler.clear_memory()
        del handler
        gc.collect()
        return zero_day_recall

    def train_on_task(self, task_idx, task_info):
        raw_task_data = task_info["data"]
        spec = task_info["spec"]
        new_labels = task_info["new_labels"]

        print(f"\n[Train] METHOD_NAME = {METHOD_NAME}")

        # 1. Expand the class set with new attack labels
        self.class_manager.add_classes(new_labels)
        current_new_ids = [self.class_manager.class_to_idx[label] for label in new_labels]
        all_known_ids = list(self.class_manager.class_to_idx.values())
        old_class_ids = [i for i in all_known_ids if i not in current_new_ids]

        # 2. Fit preprocessor on Task 0
        if task_idx == 0:
            print("Fitting Preprocessor on Task 0...")
            num_cols = set(spec.include_fields) - set(spec.categorical_fields)
            for c in num_cols:
                self.pre_processor.fit_numerical(
                    c, pd.to_numeric(raw_task_data[c], errors="coerce").fillna(0).values
                )
            for c in spec.categorical_fields:
                if c != spec.class_column:
                    self.pre_processor.fit_categorical(c, raw_task_data[c].astype(str).values)
            self.update_input_spec(raw_task_data, spec)

        # 3. Pre-update screening: dual-consistency benign drift filtering (Stage 1)
        training_data_to_use = raw_task_data.copy()
        if task_idx > 0 and USE_DRIFT_FILTER:
            print(f"\n[Drift Logic] Checking Benign samples in Task {task_idx}...")
            df_benign = raw_task_data[raw_task_data[spec.class_column] == "Benign"]
            df_attack = raw_task_data[raw_task_data[spec.class_column] != "Benign"]
            df_benign_filtered = self.filter_benign_via_drift(df_benign, spec)
            training_data_to_use = pd.concat([df_attack, df_benign_filtered], ignore_index=True)
            print(f"[Drift] ON -> {len(training_data_to_use)} samples after filtering")
        else:
            print(f"[Drift] OFF -> {len(training_data_to_use)} raw task samples")

        # 4. Clone old model for selective distillation
        self.prev_model = None
        if self.model is not None and USE_DISTILL and task_idx > 0:
            print("Cloning current model as 'Old Model' for selective distillation...")
            self.prev_model = tf.keras.models.clone_model(self.model)
            self.prev_model.set_weights(self.model.get_weights())

        # 5. Build initial model
        if self.model is None:
            print("\n>>> Building Initial Model...")
            self.model = self.model_builder_fn(self.global_input_spec)

        # 6. Representative rehearsal replay
        if USE_REPLAY:
            mem_data, mem_y_idx = self.memory.get()
            if not mem_data.empty:
                mem_data_rec = mem_data.copy()
                mem_data_rec[spec.class_column] = mem_y_idx.map(self.class_manager.idx_to_class)
                combined_train_data = pd.concat([training_data_to_use, mem_data_rec], ignore_index=True)
                print(f"[Replay] ON -> {len(training_data_to_use)} (Current) + {len(mem_data_rec)} (Memory)")
            else:
                combined_train_data = training_data_to_use
                print(f"[Replay] ON but memory empty -> {len(combined_train_data)} current samples")
        else:
            combined_train_data = training_data_to_use
            print(f"[Replay] OFF -> {len(combined_train_data)} current-task samples only")

        combined_train_data = combined_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # 7. Load dataset
        handler = FlowTransformer(self.pre_processor, **self.ft_components)
        handler.model_input_spec = self.global_input_spec
        handler.load_dataset(
            f"train_{task_idx}",
            combined_train_data,
            spec,
            self.class_manager.class_to_idx,
            evaluation_percent=0.0,
            fit_preprocessor=False,
        )

        # 8. Protected incremental update (Stage 2): BCE + SupCon + selective KD
        steps_per_epoch = max(1, int(handler.training_mask.sum()) // BATCH_SIZE)
        print(f"\n>>> Training Task {task_idx} with method [{METHOD_NAME}] ...")
        print(
            f"[Train] Samples={int(handler.training_mask.sum())}, Batch={BATCH_SIZE}, "
            f"Steps/Epoch={steps_per_epoch}"
        )
        self.model = handler.train_incremental(
            self.model,
            old_model=self.prev_model if USE_DISTILL else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            steps_per_epoch=steps_per_epoch,
            benign_class_idx=0,
            lr=LEARNING_RATE,
            distill_lambda=DISTILL_LAMBDA if (USE_DISTILL and task_idx > 0) else 0.0,
            old_class_ids=old_class_ids if USE_DISTILL else None,
            supcon_lambda=SUPCON_LAMBDA if USE_SUPCON else 0.0,
            temperature=SUPCON_TEMP,
        )

        handler.clear_memory()
        del handler
        gc.collect()

        # 9. Post-update benign profile refresh (Stage 3)
        if UPDATE_BENIGN_PROFILE:
            benign_pool = raw_task_data[raw_task_data[spec.class_column] == "Benign"]
            if len(benign_pool) > 50000:
                benign_pool = benign_pool.sample(n=50000, random_state=42)
            if not benign_pool.empty:
                self.update_benign_profile(benign_pool, spec, is_first_task=(task_idx == 0))
            print("[Benign Profile] Updated.")
        else:
            print("[Benign Profile] Skipped.")

        # 10. Post-update attack memory refresh (Stage 3)
        if UPDATE_MEMORY:
            print("\n[Memory] Updating Rehearsal Memory...")
            feature_extractor_fn = lambda df: self._extract_features(df, spec)
            current_task_y_idx_raw = training_data_to_use[spec.class_column].map(
                self.class_manager.class_to_idx
            )
            self.memory.add(training_data_to_use, current_task_y_idx_raw, self.class_manager, feature_extractor_fn)
        else:
            print("[Memory] Skipped.")

        self.task_data_info.append(task_info)

    def update_input_spec(self, data, spec):
        num = [c for c in (set(spec.include_fields) - set(spec.categorical_fields)) if c in data.columns]
        cat = [c for c in spec.categorical_fields if c != spec.class_column and c in data.columns]
        levels = [len(self.pre_processor.encoded_levels.get(c, [])) for c in cat]
        self.global_input_spec = ModelInputSpecification(
            num + cat, len(num), levels, self.ft_components["input_encoding"].required_input_format
        )

    def evaluate_comprehensive(self, current_task_idx):
        """Task-Aware / Global / New / Old evaluation with a fixed 0.5 threshold."""
        print(f"\n>>> Comprehensive Evaluation for Task {current_task_idx} <<<")
        test_pool = self.task_data_info[0]["test_pool"]
        full_test = self.task_data_info[0]["full_test"]
        spec = self.task_data_info[0]["spec"]
        current_benign_test = self.task_data_info[current_task_idx]["task_benign_test"]
        current_new_attacks = self.task_data_info[current_task_idx]["new_labels"]

        old_attacks = []
        if current_task_idx > 0:
            for i in range(current_task_idx):
                old_attacks.extend(self.task_data_info[i]["new_labels"])
        seen_attacks = old_attacks + current_new_attacks
        global_attacks = [str(lbl) for lbl in test_pool.keys() if str(lbl) != "Benign"]

        def build_eval_df(attack_labels, benign_df, mode_name):
            if mode_name == "Global":
                return full_test.copy().reset_index(drop=True)
            if not attack_labels:
                return pd.DataFrame()
            dfs = []
            if benign_df is not None and not benign_df.empty:
                dfs.append(benign_df.copy().reset_index(drop=True))
            for label in attack_labels:
                if label in test_pool and not test_pool[label].empty:
                    dfs.append(test_pool[label])
            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True).reset_index(drop=True)

        def build_eval_class_to_idx(group_name, attack_labels):
            if group_name != "Global":
                return self.class_manager.class_to_idx
            class_to_idx = copy.deepcopy(self.class_manager.class_to_idx)
            next_idx = max(class_to_idx.values(), default=-1) + 1
            for label in attack_labels:
                if label not in class_to_idx:
                    class_to_idx[label] = next_idx
                    next_idx += 1
            return class_to_idx

        metrics_summary = {}
        metrics_groups = [
            ("TaskAware", seen_attacks, current_benign_test),
            ("Global", global_attacks, None),
            ("New", current_new_attacks, current_benign_test),
            ("Old", old_attacks, current_benign_test),
        ]
        current_task_threshold = float(self.decision_threshold)

        def absorb_result(group_name, attacks, res):
            for k, v in res.items():
                if isinstance(v, (int, float, np.floating)):
                    metrics_summary[f"{k}_{group_name}"] = float(v)
                elif isinstance(v, np.ndarray):
                    metrics_summary[f"{k}_{group_name}"] = v
                elif isinstance(v, (str, bool, np.bool_)):
                    metrics_summary[f"{k}_{group_name}"] = v
            metrics_summary[f"requested_attacks_{group_name}"] = ",".join(attacks) if attacks else ""

        for group_name, attacks, benign_df in metrics_groups:
            if group_name == "Old" and current_task_idx == 0:
                continue
            df_eval = build_eval_df(attacks, benign_df, group_name)
            if df_eval.empty:
                continue

            eval_class_to_idx = build_eval_class_to_idx(group_name, attacks)

            handler = FlowTransformer(self.pre_processor, **self.ft_components)
            handler.m = self.model
            handler.model_input_spec = self.global_input_spec

            res = handler.evaluate_binary(
                f"Eval_{group_name}_Task_{current_task_idx}",
                df_eval,
                spec,
                eval_class_to_idx,
                benign_class_idx=0,
                do_resampled_eval=True,
                decision_threshold=current_task_threshold,
                search_best_threshold=False,
            )
            absorb_result(group_name, attacks, res)

            handler.clear_memory()
            del handler, df_eval
            gc.collect()

        metrics_summary["current_task_threshold"] = current_task_threshold
        return metrics_summary


# ============================================================================
# Main entry
# ============================================================================
def run_experiment(args, seed, dataset_path, results_csv=None):
    ft_components = {
        "input_encoding": RecordLevelEmbed(64),
        "sequential_model": BasicTransformer(2, 128, n_heads=2),
        "classification_head": LastTokenClassificationHead(),
        "params": FlowTransformerParameters(
            window_size=SLIDING_WINDOW, mlp_layer_sizes=[128], mlp_dropout=0.1
        ),
    }

    # Seed all randomness for this independent run
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tasks, pre_proc = prepare_data_custom(dataset_path, NamedDatasetSpecifications.unified_flow_format)

    def model_builder(spec):
        ft = FlowTransformer(pre_proc, **ft_components)
        ft.model_input_spec = spec
        ft.X = pd.DataFrame(columns=spec.feature_names)
        return ft.build_model()

    learner = IncrementalLearner(model_builder, pre_proc, ft_components)
    history = []

    for i, task in enumerate(tasks):
        print(f"\n{'#' * 30}")
        print(f" Processing Task {i}: {task['name']}")
        print(f"{'#' * 30}")

        zero_day_recall = 0.0
        if i > 0:
            zero_day_recall = learner.evaluate_zero_day(i, task)

        learner.train_on_task(i, task)

        print(f"\n>>> Evaluating Task {i} Performance (Comprehensive)...")
        res = learner.evaluate_comprehensive(i)

        record = {
            "Seed": seed,
            "Task": i,
            "Name": task["name"],
            "ZeroDay_Recall": zero_day_recall,
            "Acc_TaskAware": res.get("real_acc_TaskAware", np.nan),
            "BalAcc_TaskAware": res.get("real_bal_acc_TaskAware", np.nan),
            "F1_TaskAware": res.get("real_f1_TaskAware", np.nan),
            "Prec_TaskAware": res.get("real_prec_TaskAware", np.nan),
            "Rc_TaskAware": res.get("real_recall_TaskAware", np.nan),
            "FPR_TaskAware": res.get("real_fpr_TaskAware", np.nan),
            "PR_AUC_TaskAware": res.get("real_pr_auc_TaskAware", np.nan),
            "Acc_Global": res.get("real_acc_Global", np.nan),
            "F1_Global": res.get("real_f1_Global", np.nan),
            "Rc_Global": res.get("real_recall_Global", np.nan),
            "FPR_Global": res.get("real_fpr_Global", np.nan),
            "Rc_New": res.get("real_recall_New", np.nan),
            "Rc_Old": res.get("real_recall_Old", np.nan),
        }
        history.append(record)

        print(
            f"Result Task {i} [TaskAware]: BalAcc={record['BalAcc_TaskAware']:.4f}, "
            f"F1={record['F1_TaskAware']:.4f}, Recall={record['Rc_TaskAware']:.4f}, "
            f"FPR={record['FPR_TaskAware']:.4f}, PR-AUC={record['PR_AUC_TaskAware']:.4f}, "
            f"ZeroDay={zero_day_recall:.4f}, New_Rc={record['Rc_New']:.4f}, "
            f"Old_Rc={record['Rc_Old']:.4f}"
        )

    final_df = pd.DataFrame(history)
    cols_order = [
        "Seed", "Task", "Name", "ZeroDay_Recall",
        "Acc_TaskAware", "BalAcc_TaskAware", "F1_TaskAware", "Prec_TaskAware",
        "Rc_TaskAware", "FPR_TaskAware", "PR_AUC_TaskAware",
        "Acc_Global", "F1_Global", "Rc_Global", "FPR_Global",
        "Rc_New", "Rc_Old",
    ]
    existing_cols = [c for c in cols_order if c in final_df.columns]
    print("\n" + "=" * 30 + f" Results Summary (seed={seed}) " + "=" * 30)
    print(tabulate(final_df[existing_cols], headers="keys", tablefmt="grid", floatfmt=".4f"))

    if results_csv is not None:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not results_csv.exists() or results_csv.stat().st_size == 0
        final_df.to_csv(results_csv, mode="a", header=write_header, index=False)
        print(f"[Output] Appended {len(final_df)} task rows to {results_csv}")

    return final_df


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        raise SystemExit(2)

    seeds = list(dict.fromkeys(args.seeds))
    summary_csv = Path(args.summary_csv) if args.summary_csv else None

    print(
        f"[Experiment] method={METHOD_NAME}, seeds={seeds}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, window={SLIDING_WINDOW}, "
        f"lambda_sc={SUPCON_LAMBDA}, lambda_kd={DISTILL_LAMBDA}, "
        f"delta=({CONFIDENCE_THRESHOLD},{DRIFT_ATTACK_PROB_UPPER})"
    )

    # Override if provided via CLI (defaults are the paper values)
    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    all_runs = []
    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f" RUN with seed={seed}")
        print(f"{'#' * 60}")
        run_df = run_experiment(args, seed, args.dataset, results_csv=summary_csv)
        all_runs.append(run_df)

    if len(all_runs) > 1:
        combined = pd.concat(all_runs, ignore_index=True)
        metric_cols = [
            "ZeroDay_Recall", "Acc_TaskAware", "BalAcc_TaskAware", "F1_TaskAware",
            "Prec_TaskAware", "Rc_TaskAware", "FPR_TaskAware", "PR_AUC_TaskAware",
            "Acc_Global", "F1_Global", "Rc_Global", "FPR_Global", "Rc_New", "Rc_Old",
        ]
        print("\n" + "=" * 30 + " Aggregated Results (mean +/- std) " + "=" * 30)
        rows = []
        for task in sorted(combined["Task"].unique()):
            sub = combined[combined["Task"] == task]
            row = {"Task": task}
            for m in metric_cols:
                if m in sub.columns:
                    vals = sub[m].dropna().astype(float)
                    if len(vals):
                        row[f"{m}_mean"] = vals.mean()
                        row[f"{m}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            rows.append(row)
        agg = pd.DataFrame(rows)
        print(tabulate(agg, headers="keys", tablefmt="grid", floatfmt=".4f"))


if __name__ == "__main__":
    main()
