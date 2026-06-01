# @Time : 2025/12/9 20:46
# @File : main_nb15_candid_methods.py
# @Author : gly
# main_fitday1.py
# NB15 version: Dual-Consistency Drift Detection + Zero-Day Evaluation + Hybrid Incremental Learning
# This file is intentionally separate from main_ids2018_3.10_dynamic_22_changge.py.

import os
import copy
import gc
import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Framework Imports
from candid_lib.engine_unsw_nb15 import FlowTransformer
from candid_lib.dataset_specification import NamedDatasetSpecifications
from candid_lib.flow_transformer_parameters import FlowTransformerParameters
from candid_lib.model_input_specification import ModelInputSpecification
from candid_lib.preprocessing import StandardPreProcessing
from candid_lib.basic_transformers import BasicTransformer
from candid_lib.classification_heads import LastTokenClassificationHead
from candid_lib.input_encodings import RecordLevelEmbed


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ================== Hyperparameters ==================
INCREMENTAL_EPOCHS = 1
INCREMENTAL_BATCH_SIZE = 512
MEMORY_PER_CLASS = 200  # Replay Buffer per Attack Class
BENIGN_MEMORY_SIZE = 2000  # Replay Buffer for Benign
FEATURE_EXTRACTION_BATCH_SIZE = 1024
MAX_MEMORY_CANDIDATES_PER_ATTACK = 500
MAX_MEMORY_CANDIDATES_BENIGN = 2000
ORIGINAL_EVAL_PERCENT = 0.2  # Per-class / per-group evaluation split ratio
EVAL_SPLIT_MODE = "stratified"  # "stratified", "last", or "random"
EVAL_RANDOM_SEED = 42
# Loss weights used by the CANDID update objective.
DISTILL_LAMBDA = 5.0
SUPCON_LAMBDA = 0.1
SUPCON_TEMP = 0.07

# Consistency Check Thresholds
CONFIDENCE_THRESHOLD = 0.3  # 闁告劕纾悺銉х矚濞差亝锛熼梻鍐ㄧ墕?
# ================== CANDID switches ==================
METHOD_NAME = "CANDID"
USE_REPLAY = True
USE_DISTILL = True
USE_SUPCON = True
USE_DRIFT_FILTER = True
UPDATE_MEMORY = True
UPDATE_BENIGN_PROFILE = True
DRIFT_FILTER_MODE = "dual"





# ================== Helper Functions ==================

def group_attack_labels_unsw(label):
    s_label = str(label).strip().lower()
    if 'normal' in s_label or 'benign' in s_label: return 'Benign'
    if 'fuzzer' in s_label: return 'Fuzzers'
    if 'analysis' in s_label: return 'Analysis'
    if 'backdoor' in s_label: return 'Backdoor'
    if 'dos' in s_label: return 'DoS'
    if 'exploit' in s_label: return 'Exploits'
    if 'generic' in s_label: return 'Generic'
    if 'reconnaissance' in s_label: return 'Reconnaissance'
    if 'shellcode' in s_label: return 'Shellcode'
    if 'worm' in s_label: return 'Worms'
    return 'Generic'


def group_attack_labels_2018(label):
    # 1. 濡澘瀚ˇ鈺呮偠閸☆厾绐楅弶鐑嗗墮閻⊙呯箔閿旇儻顩柨娑樿嫰楠炴挾绮氶悜妯煎闁挎稑鐭佸ù鍡欎焊韫囨挸?(閻庣懓鑻崣蹇旂鐠恒劌?NB15 闁告劖鐟︾涵?
    s_label = str(label).strip().lower()

    # 2. Benign (婵繐绲介悥璺好规笟鈧崳?
    if 'benign' in s_label: return 'Benign'

    # 3. DDoS (婵炲鍔嶉崜浼存晬濮橆剛绠戝銈囩帛閺備線?DoS 濞戞柨顑呮晶鐘绘晬?
    # 闁搞儳濮崇拹鐔衡偓娑欘殘椤戜焦?'ddos' 濞戞搩鍘肩€垫﹢?'dos'闁挎稑鑻々褔寮稿鍐ㄥ弗闁告帇鍊栭弻?dos闁挎稑鐣砫os 闁轰胶澧楀畵浣瑰濮樻剚娼堕悹鍥跺灠?
    if 'ddos' in s_label: return 'DDoS'

    # 4. DoS
    if 'dos' in s_label: return 'DoS'

    # 5. Web Attacks
    # 闁搞儱澧芥晶鏍箵韫囨艾鐗氬☉鎿冨幗瑜颁線宕氶弶鍨樁?SQL injections, command injections ?
    if 'web' in s_label or 'sql' in s_label or 'xss' in s_label or 'injection' in s_label: return 'Web Attacks'

    # 6. BruteForce
    # 閻庣數鎳撶花鏌ュ炊閸撗冾暬濞戞搩鍘惧▓?BruteForce闁挎稑鐭傞埀顒佽壘閻栧爼宕㈤悢濂夋綏闁哄秴娲ㄩ鐑藉礌閸涱厽?'brute', 'ssh', 'ftp' ?
    if 'brute' in s_label or 'ssh' in s_label or 'ftp' in s_label: return 'BruteForce'

    # 7. Infiltration
    # 濞ｅ浂鍠楅婊勭閸℃鏂уù鐙呯悼閻栨粍绋夐鐘崇暠 'infilteration' 闁归鍘ч崯鎾绘煥濞嗘帩?
    if 'infilteration' in s_label: return 'Infiltration'

    # 8. Bot
    if 'bot' in s_label: return 'Bot'

    # 9. 闁稿繑绮岀花瀹犮亹閹烘棁?(闁惧繒鏅崝褔宕堕崜褍顣诲☉鎿冨幗閸忔粓鎯勯弽锔惧晩闁圭鍋撻柡鍫濐檧缁辨繃鎷呴崱鏇犵闁伙絾鐟ゆ禍鎺楁⒓闊厾顏诲☉鎾亾)
    return 'Other'



def group_attack_labels(label):
    """Groups detailed attack labels into broader categories."""
    if pd.isna(label): return 'Other_Attack'
    s_label = str(label).lower()
    if 'benign' in s_label: return 'Benign'
    if 'ddos' in s_label: return 'DDoS'
    if 'dos' in s_label: return 'DoS'
    if 'web' in s_label or 'sql' in s_label or 'xss' in s_label: return 'WebAttack'
    if 'brute' in s_label: return 'BruteForce'
    if 'infilteration' in s_label: return 'Infiltration'
    if 'bot' in s_label: return 'Bot'
    return 'Other_Attack'









def prepare_data_custom(dataset_path, dataset_spec):
    print("=" * 20 + " Data Preparation (Attack Stratified + Benign Time Grouping) " + "=" * 20)
    df_full = pd.read_csv(dataset_path)

    if 'attack_cat' in df_full.columns:
        dataset_spec.class_column = 'attack_cat'

    time_columns = [c for c in ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS'] if c in df_full.columns]
    if time_columns:
        for col_name in time_columns:
            df_full[col_name] = pd.to_numeric(df_full[col_name], errors='coerce')
        print(f"Detected benign grouping time columns: {time_columns}")
    else:
        print("Warning: time columns not found, benign grouping will follow the current file order.")

    df_full['Grouped_Label'] = df_full[dataset_spec.class_column].apply(group_attack_labels_unsw)
    spec_grouped = copy.deepcopy(dataset_spec)
    spec_grouped.class_column = 'Grouped_Label'
    split_mode = str(EVAL_SPLIT_MODE).strip().lower()

    # base_attacks = ['DoS', 'Web Attacks', 'BruteForce']
    # inc_attacks = ['DDoS','Infiltration', 'Bot']




    #  鍘熸湰
    # base_attacks = ['DoS', 'DDoS','Infiltration' ]
    # inc_attacks = ['Web Attacks','BruteForce' ,'Bot']


    grouped_labels = df_full['Grouped_Label'].unique()
    available_labels = set(grouped_labels)
    base_attacks = [a for a in ['Fuzzers', 'Analysis', 'Backdoor', 'DoS', 'Reconnaissance', 'Shellcode'] if a in available_labels]
    inc_attacks = [a for a in ['Worms', 'Exploits', 'Generic'] if a in available_labels]
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
                df_subset = df_subset.sort_values(by=sort_columns, kind='stable').reset_index(drop=True)
            split_at = n_total - n_test
            return (
                df_subset.iloc[:split_at].copy().reset_index(drop=True),
                df_subset.iloc[split_at:].copy().reset_index(drop=True)
            )

        if mode in {"random", "stratified"}:
            shuffled = df_subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            split_at = n_total - n_test
            return (
                shuffled.iloc[:split_at].copy().reset_index(drop=True),
                shuffled.iloc[split_at:].copy().reset_index(drop=True)
            )

        raise ValueError(f"Unsupported split mode: {mode}")

    if split_mode == "stratified":
        print(
            f"Evaluation split mode: stratified per attack class + chronological benign groups "
            f"(test_ratio={ORIGINAL_EVAL_PERCENT:.0%}, seed={EVAL_RANDOM_SEED})"
        )
        pool_train = {}
        pool_test = {}

        benign_all = df_full[df_full['Grouped_Label'] == 'Benign'].copy().reset_index(drop=True)
        if benign_all.empty:
            raise ValueError("No Benign samples found in the dataset.")
        if time_columns:
            benign_all = benign_all.sort_values(by=time_columns, kind='stable').reset_index(drop=True)
            print(f"Benign samples sorted by {time_columns} before chronological grouping.")
        else:
            print("Warning: benign chronological grouping falls back to current file order.")

        benign_chunks_full = [
            chunk.reset_index(drop=True)
            for chunk in np.array_split(benign_all, total_tasks)
            if len(chunk) > 0
        ]
        if len(benign_chunks_full) < total_tasks:
            raise ValueError(
                f"Benign chronological groups are insufficient: expected {total_tasks}, got {len(benign_chunks_full)}."
            )

        benign_train_groups = []
        benign_test_groups = []
        for idx, chunk in enumerate(benign_chunks_full):
            train_chunk, test_chunk = split_dataframe_for_eval(
                chunk,
                mode="last",
                random_state=EVAL_RANDOM_SEED + idx,
                test_ratio=ORIGINAL_EVAL_PERCENT,
                sort_columns=None
            )
            benign_train_groups.append(train_chunk)
            benign_test_groups.append(test_chunk)
            print(
                f"[Benign Group {idx}] train={len(train_chunk)}, test={len(test_chunk)}"
            )

        pool_train['Benign'] = pd.concat(benign_train_groups, ignore_index=True).reset_index(drop=True)
        pool_test['Benign'] = pd.concat(benign_test_groups, ignore_index=True).reset_index(drop=True)

        for cls_idx, cls in enumerate(grouped_labels):
            if cls == 'Benign':
                continue
            df_cls = df_full[df_full['Grouped_Label'] == cls].copy().reset_index(drop=True)
            train_cls, test_cls = split_dataframe_for_eval(
                df_cls,
                mode="stratified",
                random_state=EVAL_RANDOM_SEED + 100 + cls_idx,
                test_ratio=ORIGINAL_EVAL_PERCENT
            )
            pool_train[cls] = train_cls
            pool_test[cls] = test_cls
            print(f"[Attack Split] {cls}: train={len(train_cls)}, test={len(test_cls)}")

        df_train = pd.concat(pool_train.values(), ignore_index=True).reset_index(drop=True)
        df_test = pd.concat(pool_test.values(), ignore_index=True).reset_index(drop=True)
    else:
        split_idx = int(len(df_full) * (1.0 - ORIGINAL_EVAL_PERCENT))
        if split_mode == "last":
            df_train = df_full.iloc[:split_idx].copy().reset_index(drop=True)
            df_test = df_full.iloc[split_idx:].copy().reset_index(drop=True)
            print(
                f"Evaluation split mode: LastRows {ORIGINAL_EVAL_PERCENT:.0%} "
                f"-> train={len(df_train)}, test={len(df_test)}"
            )
        elif split_mode == "random":
            shuffled_df = df_full.sample(frac=1.0, random_state=EVAL_RANDOM_SEED).reset_index(drop=True)
            df_train = shuffled_df.iloc[:split_idx].copy().reset_index(drop=True)
            df_test = shuffled_df.iloc[split_idx:].copy().reset_index(drop=True)
            print(
                f"Evaluation split mode: Random {ORIGINAL_EVAL_PERCENT:.0%} (seed={EVAL_RANDOM_SEED}) "
                f"-> train={len(df_train)}, test={len(df_test)}"
            )
        else:
            raise ValueError(f"Unsupported EVAL_SPLIT_MODE: {EVAL_SPLIT_MODE}. Use 'stratified', 'last' or 'random'.")

        pool_train = {
            cls: df_train[df_train['Grouped_Label'] == cls].copy().reset_index(drop=True)
            for cls in grouped_labels
        }
        pool_test = {
            cls: df_test[df_test['Grouped_Label'] == cls].copy().reset_index(drop=True)
            for cls in grouped_labels
        }

    base_attacks = [a for a in base_attacks if a in pool_train and not pool_train[a].empty]
    inc_attacks = [a for a in inc_attacks if a in pool_train and not pool_train[a].empty]
    total_tasks = 1 + len(inc_attacks)

    if 'Benign' not in pool_train or pool_train['Benign'].empty:
        raise ValueError("No Benign samples found in the training split.")

    if split_mode == "stratified":
        benign_train_groups = [chunk.reset_index(drop=True) for chunk in benign_train_groups[:total_tasks]]
        benign_test_groups = [chunk.reset_index(drop=True) for chunk in benign_test_groups[:total_tasks]]
    else:
        df_benign_all = pool_train['Benign'].copy().reset_index(drop=True)
        if time_columns:
            df_benign_all = df_benign_all.sort_values(by=time_columns, kind='stable')
            print(f"Training benign samples sorted by {time_columns} before grouping.")
        df_benign_all = df_benign_all.reset_index(drop=True)
        benign_train_groups = [
            chunk.reset_index(drop=True)
            for chunk in np.array_split(df_benign_all, total_tasks)
            if len(chunk) > 0
        ]
        df_benign_test_all = pool_test['Benign'].copy().reset_index(drop=True)
        if time_columns:
            df_benign_test_all = df_benign_test_all.sort_values(by=time_columns, kind='stable')
        benign_test_groups = [
            chunk.reset_index(drop=True)
            for chunk in np.array_split(df_benign_test_all, total_tasks)
            if len(chunk) > 0
        ]

    if len(benign_train_groups) < total_tasks or len(benign_test_groups) < total_tasks:
        raise ValueError(
            f"Benign task groups are insufficient after split: train_groups={len(benign_train_groups)}, "
            f"test_groups={len(benign_test_groups)}, expected={total_tasks}."
        )

    def summarize_label_counts(df, label_col):
        if df is None or df.empty:
            return {}
        vc = df[label_col].value_counts(dropna=False).sort_index()
        return {str(k): int(v) for k, v in vc.items()}

    def print_data_audit():
        print("\n" + "=" * 20 + " Data Audit Summary " + "=" * 20)
        print("[Audit] Attack Class Train/Test Counts")
        for label in base_attacks + inc_attacks:
            train_n = len(pool_train.get(label, pd.DataFrame()))
            test_n = len(pool_test.get(label, pd.DataFrame()))
            print(f"  - {label}: train={train_n}, test={test_n}")

        print("[Audit] Benign Group Train/Test Counts")
        for idx, (train_group, test_group) in enumerate(zip(benign_train_groups, benign_test_groups)):
            print(f"  - B{idx}: train={len(train_group)}, test={len(test_group)}")

        print("[Audit] Task Train Composition")
        for task in tasks:
            train_counts = summarize_label_counts(task['data'], spec_grouped.class_column)
            print(
                f"  - {task['name']}: benign_group=B{task['benign_group_idx']}, "
                f"new_labels={task['new_labels']}, train_labels={train_counts}"
            )

        print("[Audit] Task TaskAware Test Composition")
        seen_attacks = []
        for task in tasks:
            seen_attacks.extend(task['new_labels'])
            benign_df = task['task_benign_test']
            dfs = [benign_df.copy().reset_index(drop=True)] if benign_df is not None and not benign_df.empty else []
            for label in seen_attacks:
                if label in pool_test and not pool_test[label].empty:
                    dfs.append(pool_test[label].copy().reset_index(drop=True))
            taskaware_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True) if dfs else pd.DataFrame()
            taskaware_counts = summarize_label_counts(taskaware_df, spec_grouped.class_column)
            print(
                f"  - {task['name']}: benign_group=B{task['benign_group_idx']}, "
                f"seen_attacks={seen_attacks}, taskaware_labels={taskaware_counts}"
            )

        global_counts = summarize_label_counts(df_test, spec_grouped.class_column)
        print(f"[Audit] Global Test Composition: {global_counts}")

    tasks = []

    def build_imbalanced_task(attack_names, task_name, benign_group_idx):
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
            "full_test": df_test.copy()
        }

    next_benign_group_idx = 0
    if base_attacks:
        print(f"Building Task 0 (Benign Group {next_benign_group_idx})...")
        t0 = build_imbalanced_task(base_attacks, "Task_0_Base", benign_group_idx=next_benign_group_idx)
        if t0:
            tasks.append(t0)
            next_benign_group_idx += 1
    else:
        print("Warning: no base attacks available in the training split; Task 0 will be skipped.")

    for att in inc_attacks:
        task_idx = len(tasks)
        benign_group_idx = next_benign_group_idx
        print(f"Building Incremental Task for {att} (Benign Group {benign_group_idx})...")
        ti = build_imbalanced_task([att], f"Task_{task_idx}_{att}", benign_group_idx=benign_group_idx)
        if ti: tasks.append(ti)
        next_benign_group_idx += 1

    print_data_audit()

    return tasks, StandardPreProcessing(n_categorical_levels=32)


class ClassManager:
    def __init__(self):
        self.class_to_idx = {'Benign': 0}
        self.idx_to_class = {0: 'Benign'}
        self.known_classes = 1

    def add_classes(self, labels):
        for l in labels:
            if l not in self.class_to_idx and l != 'Benign':
                idx = self.known_classes
                self.class_to_idx[l] = idx
                self.idx_to_class[idx] = l
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
        """Select representative replay samples with normalized feature vectors."""
        if len(df_candidates) <= max_size:
            return df_candidates
        max_candidates = max(
            max_size,
            MAX_MEMORY_CANDIDATES_BENIGN if max_size == self.benign_memory_size else MAX_MEMORY_CANDIDATES_PER_ATTACK
        )
        df_candidates = self._downsample_candidates(df_candidates, max_candidates)

        # 1. 闁圭粯鍔曡ぐ鍥磹濞嗘挴鍋撴径瀣闁哄牜鍓涘▓鎴︽偋閻熸壆?
        feats = feature_extractor(df_candidates)
        if len(feats) == 0:
            return df_candidates.sample(n=max_size, random_state=42)

        # 2. 閻犱緤绱曢悾濠氭偋閻熸壆绐欓悹鎰╁妼?(Centroid) 妤犵偠缈伴埀顒佸姇缁烩偓濡炪倖鐪归埀顒佸灱缁绘?L2 鐟滅増甯婄粩鎾礌?
        centroid = np.mean(feats, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

        # 3. 濞达綀娉曢弫銈嗘媴濞嗗繘娴悹鐑樼箘?(1 - Cosine Similarity) 闁哄洤銇橀崬顒€鈻庤閻剛鎹勫┑鍫?
        sims = np.dot(feats, centroid)
        distances = 1.0 - sims

        # 4. 闂侇偄顦扮€氥劎鎹勫┑鍫€查悹鎰╁妼缁洪箖寮甸埀顒佹交閹寸姵?max_size 濞戞搩浜濋悧閬嶅嫉?
        closest_indices = np.argpartition(distances, max_size - 1)[:max_size]
        closest_indices = closest_indices[np.argsort(distances[closest_indices])]

        return df_candidates.iloc[closest_indices].reset_index(drop=True)

    def add(self, new_data: pd.DataFrame, new_labels_idx: pd.Series, class_manager: ClassManager, feature_extractor):
        num_known_classes = class_manager.known_classes
        if num_known_classes == 0: return

        final_data_list = []
        final_labels_list = []

        # 1. Benign (Label 0)
        current_benign_mask = (new_labels_idx == 0)
        old_benign_mask = (self.labels == 0)

        candidates_benign = pd.concat([
            self.data.loc[old_benign_mask] if not self.data.empty else pd.DataFrame(),
            new_data.loc[current_benign_mask]
        ], ignore_index=True)

        if not candidates_benign.empty:
            kept_benign = self._select_representative(candidates_benign, self.benign_memory_size, feature_extractor)
            final_data_list.append(kept_benign)
            final_labels_list.append(pd.Series([0] * len(kept_benign)))

        # 2. Attacks (Label > 0)
        for class_id in range(1, num_known_classes):
            new_mask = (new_labels_idx == class_id)
            old_mask = (self.labels == class_id)

            candidates_attack = pd.concat([
                self.data.loc[old_mask] if not self.data.empty else pd.DataFrame(),
                new_data.loc[new_mask]
            ], ignore_index=True)

            if not candidates_attack.empty:
                kept_attack = self._select_representative(candidates_attack, self.memory_per_class, feature_extractor)
                final_data_list.append(kept_attack)
                final_labels_list.append(pd.Series([class_id] * len(kept_attack)))

        if final_data_list:
            self.data = pd.concat(final_data_list, ignore_index=True)
            self.labels = pd.concat(final_labels_list, ignore_index=True)

    def get(self):
        return self.data, self.labels


# ==============================================================================
# 濞ｅ浂鍠栭ˇ鏌ユ偋?Incremental Learner (闁告瑦鐗楃粔鐑芥煥濞嗘帩鍤栭柣銊ュ閳规牠姊荤€涙绌块柛姘墳缁辨繃鏅堕悙鎻掝潱闂傚啯褰冪粚鍧楁儎濮椻偓閺€?
# ==============================================================================
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

    def _extract_features(self, df_data, spec):
        if df_data.empty or self.model is None: return np.array([])

        temp_handler = FlowTransformer(self.pre_processor, **self.ft_components)
        temp_handler.model_input_spec = self.global_input_spec
        df, _ = temp_handler._load_preprocessed_dataset(
            "feature_extraction", df_data, spec, fit_preprocessor=False, evaluation_percent=0.0
        )
        if "__training" in df.columns: del df["__training"]
        if "__y" in df.columns: del df["__y"]

        X_np = df.values.astype(np.float32)
        X_np_padded = np.vstack([X_np, np.zeros((1, X_np.shape[1]), dtype=np.float32)])
        padding_idx = X_np.shape[0]
        batch_size = FEATURE_EXTRACTION_BATCH_SIZE
        features_list = []
        W = self.ft_components['params'].window_size
        window_offsets = np.arange(-W + 1, 1)

        for i in range(0, len(df), batch_size):
            end = min(i + batch_size, len(df))
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
        print("\n[Drift Updating] Recalculating Benign Profile with latest model...")
        feats = self._extract_features(benign_data, spec)
        if len(feats) == 0: return

        # 濞ｅ浂鍠栭ˇ鏌ユ晬濮樿鲸绾柟鎭掑劙婵炲洭鎮介妸锔戒粯闁哄倹澹嗗▓鎴犵矚濞差亝锛熷☉鎿冨幖缁洪箖鏁嶇仦鑺ユ笎闁告劕鍘栫粭澶愬椽鐏炶棄鍧婇柛娆掑紦閼垫垼绠涢崘銊ょ驳 EMA 婵烇絽鍢查幃?
        new_center = np.mean(feats, axis=0)
        self.benign_center = new_center / (np.linalg.norm(new_center) + 1e-9)

        # 閻犱緤绱曢悾濠氬棘閹殿喗?IQR 闂傚啫鐗嗛埀?(EVT)
        sims = np.dot(feats, self.benign_center)
        dists = 1.0 - sims
        Q1, Q3 = np.percentile(dists, 25), np.percentile(dists, 75)
        new_threshold = Q3 + 1.5 * (Q3 - Q1)

        if is_first_task or self.benign_threshold == 0.0:
            self.benign_threshold = new_threshold
            print(f"[Drift Profile] Initialized -> Q1: {Q1:.4f} | Q3: {Q3:.4f}")
            print(f"[Drift Profile] Initial Threshold set to: {self.benign_threshold:.4f}")
        else:
            # 濞ｅ浂鍠栭ˇ鏌ユ晬濮橆剙娑ч悗浣冾潐閻栵綁鏌岃箛娑欘潓闁稿﹦鍘ф禒娑㈢嵁閾忣偆鎷ㄦ繝鏇熸煥鐎垫煡鏁嶅畝鍕╂慨婵勫灲濡插洭宕愰懖鈺冨磹?
            alpha = 0.7
            old_threshold = self.benign_threshold
            self.benign_threshold = alpha * self.benign_threshold + (1 - alpha) * new_threshold
            print(
                f"[Drift Profile] Updated Threshold (EMA) -> Old: {old_threshold:.4f} | New: {self.benign_threshold:.4f}")

    def filter_benign_via_drift(self, new_benign_data, spec):
        if self.benign_center is None:
            return new_benign_data

        print(f"\n[Drift Detection] Consistency Check on {len(new_benign_data)} samples...")

        model_to_use = self.prev_model if self.prev_model is not None else self.model
        temp_handler = FlowTransformer(self.pre_processor, **self.ft_components)
        temp_handler.model_input_spec = self.global_input_spec
        df, _ = temp_handler._load_preprocessed_dataset(
            "filtering", new_benign_data, spec, fit_preprocessor=False, evaluation_percent=0.0
        )
        if "__training" in df.columns: del df["__training"]
        if "__y" in df.columns: del df["__y"]

        X_np = df.values.astype(np.float32)
        X_np_padded = np.vstack([X_np, np.zeros((1, X_np.shape[1]), dtype=np.float32)])
        padding_idx = X_np.shape[0]
        batch_size = 512
        features_list, probs_list = [], []

        W = self.ft_components['params'].window_size
        window_offsets = np.arange(-W + 1, 1)

        for i in range(0, len(df), batch_size):
            end = min(i + batch_size, len(df))
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

        if not features_list: return new_benign_data

        feats = np.vstack(features_list)
        probs = np.vstack(probs_list).reshape(-1)

        sims = np.dot(feats, self.benign_center)
        dists = 1.0 - sims

        # 濞ｅ浂鍠栭ˇ鏌ユ晬濮樿埖些鐎规洍鏅滅花婵嬫煥?(Safety Clamp)闁靛棗鍊搁々褔寮稿鍡╁悁缂佺姵顨呴崵顓犳惥閸涙壆?30% 闁汇劌瀚弳鐔煎箲椤旇姤笑婵犳洖鍊讳簺闁挎稑鐭侀鈺呭及鎼达絺鏁勯梻鍌涙綑瑜板宕濋妸銉ｄ喊濠㈠爢宥囩鐎殿喖鎼崺妤呭箣椤忓懏?
        drift_ratio = np.mean(dists > self.benign_threshold)
        if drift_ratio > 0.3:
            adjusted_threshold = np.percentile(dists, 70)  # 鐎殿喖鎼崺妤佺┍濠靛牊娈岄柛?70%
            is_feature_drift = (dists > adjusted_threshold)
            print(
                f"  [Warning] Huge drift detected ({drift_ratio:.1%}). Clamping threshold to {adjusted_threshold:.4f}")
        else:
            is_feature_drift = (dists > self.benign_threshold)

        is_model_uncertain = (probs > CONFIDENCE_THRESHOLD)

        consistent_drift = is_feature_drift & is_model_uncertain
        simple_drift = is_feature_drift

        if DRIFT_FILTER_MODE == "feature_only":
            final_mask = simple_drift
        elif DRIFT_FILTER_MODE == "dual":
            final_mask = consistent_drift
            random_mask = (np.random.rand(len(dists)) < 0.1)
            final_mask = final_mask | random_mask
        else:
            raise ValueError(f"Unsupported DRIFT_FILTER_MODE: {DRIFT_FILTER_MODE}")

        # =================================================================
        # [闁哄秶顭堢缓鐐┍椤旂⒈妲籡 闂傚啯鐓″ú鈺佺暦閳哄懏鏁氶柨娑欏哺閹凤紕?IDS2018 閺夆晜鐟ч～鎺楀础閸愌咁伝缂佺嫏鍕闁硅鍣ｅ▔锕傛儍閸曨亞绠介柟韬插€х槐?
        # 濠碘€冲€归悘澶愬箮婵犲倸娈ら柛妞炬缁旀儳顭抽崒婧炩晠鎳濋娑掑亾瑜嶆慨鐐存交濞戞粠鍞茬紓浣稿暣濞夛箓鏁嶇仦鑲╃獥闁活剦鍓熷Λ鍨庣憴鍕⒕闁告垹濮村畷鍫ュ级閿涘嫭鐣遍柡鍌涙緲閻庣兘寮ㄧ拠鎻掓瘖 (?Bot)?
        # 鐎殿喖鎼崺妤冩喆閸曨偆鏆伴柨娑欑閻︹€斥枎?Task 闁哄牃鍋撳鑸佃壘瑜把囧礂娴ｇ瓔?30,000 闁哄銈咁棇闁诡儸鍕＝缂佸绮弳鐔煎箲椤旂厧妫樺☉鎾抽椤ゅ啴鏌岃箛鎾茬俺閻犲鍟ㄩ埀?
        # =================================================================
        MAX_DRIFT_KEEP = 30000
        n_kept_current = np.sum(final_mask)
        if n_kept_current > MAX_DRIFT_KEEP:
            true_indices = np.where(final_mask)[0]
            keep_idx = np.random.choice(true_indices, size=MAX_DRIFT_KEEP, replace=False)
            final_mask = np.zeros_like(final_mask, dtype=bool)
            final_mask[keep_idx] = True
            print(f"  [Safety Clamp] Massive drift detected ({n_kept_current}). Capped to {MAX_DRIFT_KEEP} to prevent overwhelming attacks.")

        # 4. 闁轰椒鍗抽崳鐑樼┍濠靛?(濞戞挸顑夊?
        MIN_SAMPLES = 2000
        n_kept = np.sum(final_mask)
        if n_kept < MIN_SAMPLES and len(new_benign_data) > MIN_SAMPLES:
            fallback_pool = np.where(simple_drift & ~final_mask)[0]
            needed = MIN_SAMPLES - n_kept
            if len(fallback_pool) > 0:
                add_idx = np.random.choice(fallback_pool, size=min(len(fallback_pool), needed), replace=False)
                final_mask[add_idx] = True

        n_kept = np.sum(final_mask)
        if n_kept < MIN_SAMPLES and len(new_benign_data) > MIN_SAMPLES:
            discarded_indices = np.where(~final_mask)[0]
            needed = MIN_SAMPLES - n_kept
            if len(discarded_indices) >= needed:
                boost_indices = np.random.choice(discarded_indices, size=needed, replace=False)
                final_mask[boost_indices] = True

        filtered_data = new_benign_data.iloc[final_mask].reset_index(drop=True)

        print(f"[Drift Stats] Total: {len(new_benign_data)}")
        print(f"  -> Feature Drift (Dynamic EVT): {np.sum(is_feature_drift)}")
        print(f"  -> Model Uncertain (Conf): {np.sum(is_model_uncertain)}")
        print(f"  -> Consistent Drift (Both): {np.sum(consistent_drift)}")
        print(f"  -> Final Kept for Training: {len(filtered_data)}")

        del temp_handler, df, X_np, X_np_padded
        gc.collect()
        return filtered_data

    def evaluate_zero_day(self, task_idx, task_info):
        if task_idx == 0:
            return 0.0

        new_labels = task_info['new_labels']
        print(f"\n>>> [Zero-Day Check] Testing model defense against NEW attacks: {new_labels} (Before Training) <<<")

        test_pool = task_info['test_pool']
        dfs = []
        for label in new_labels:
            if label in test_pool and not test_pool[label].empty:
                dfs.append(test_pool[label])

        if not dfs:
            print("[Zero-Day Check] No test data available for new attacks.")
            return 0.0

        df_attack_only = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        spec = task_info['spec']
        eval_class_to_idx = copy.deepcopy(self.class_manager.class_to_idx)
        next_idx = max(eval_class_to_idx.values(), default=-1) + 1
        for label in new_labels:
            if label not in eval_class_to_idx:
                eval_class_to_idx[label] = next_idx
                next_idx += 1

        handler = FlowTransformer(self.pre_processor, **self.ft_components)
        handler.m = self.model
        handler.model_input_spec = self.global_input_spec

        handler.load_dataset(f"zeroday_eval_{task_idx}", df_attack_only, spec,
                             eval_class_to_idx, evaluation_percent=1.0, fit_preprocessor=False)

        if handler.X is None or len(handler.X) == 0: return 0.0

        yielder = handler.get_batch_yielder(512, for_training=False, balanced=False)
        y_prob_list = []

        while True:
            bx, _ = yielder.get_batch()
            if not bx: break
            outputs = self.model.predict(bx, verbose=0)
            prob = outputs[0] if isinstance(outputs, list) else outputs
            y_prob_list.extend(np.array(prob).reshape(-1))

        y_prob = np.array(y_prob_list)
        preds = (y_prob > 0.5).astype(int)
        zero_day_recall = np.mean(preds)

        print(f"[Zero-Day Result] Model intercepted {np.sum(preds)}/{len(preds)} unseen attacks.")
        print(f"[Zero-Day Result] ZeroDay_Recall: {zero_day_recall:.4f}")

        handler.clear_memory()
        del handler
        gc.collect()

        return zero_day_recall

    def train_on_task(self, task_idx, task_info):
        raw_task_data = task_info['data']
        spec = task_info['spec']
        new_labels = task_info['new_labels']

        print(f"\n[Train] METHOD_NAME = {METHOD_NAME}")

        # 1. 婵炲鍔岄崬鍊熴亹閹惧啿顤呭ù鐘侯嚙婵喖鎯冮崟顒佺厐缂侇偉顕ч崺?
        self.class_manager.add_classes(new_labels)
        current_new_ids = [self.class_manager.class_to_idx[l] for l in new_labels]
        all_known_ids = list(self.class_manager.class_to_idx.values())
        old_class_ids = [i for i in all_known_ids if i not in current_new_ids]

        # 濮掓稒顭堥鑽ゆ媼椤撶姷鐭婇柡浣哄?= 鐟滅増鎸告晶鐘崇鐠囨彃顫ら柛妯煎枎椤劙寮悧鍫濈ウ
        training_data_to_use = raw_task_data.copy()

        # 2. Task 0 闁归鍠庨幃搴紣閸曨偒妲遍柣鐐叉?
        if task_idx == 0:
            print("Fitting Preprocessor on Task 0...")
            num_cols = set(spec.include_fields) - set(spec.categorical_fields)
            for c in num_cols:
                self.pre_processor.fit_numerical(
                    c,
                    pd.to_numeric(raw_task_data[c], errors='coerce').fillna(0).values
                )
            for c in spec.categorical_fields:
                if c != spec.class_column:
                    self.pre_processor.fit_categorical(c, raw_task_data[c].astype(str).values)
            self.update_input_spec(raw_task_data, spec)

        # 3. Drift Filter
        if task_idx > 0 and USE_DRIFT_FILTER:
            print(f"\n[Drift Logic] Checking Benign samples in Task {task_idx}...")
            df_benign = raw_task_data[raw_task_data[spec.class_column] == 'Benign']
            df_attack = raw_task_data[raw_task_data[spec.class_column] != 'Benign']

            df_benign_filtered = self.filter_benign_via_drift(df_benign, spec)
            training_data_to_use = pd.concat([df_attack, df_benign_filtered], ignore_index=True)

            print(f"[Drift] ON -> {len(training_data_to_use)} samples after filtering")
        else:
            print(f"[Drift] OFF -> {len(training_data_to_use)} raw task samples")

        # 4. 闁哄瀚伴埀?old model闁挎稑鐗呯划搴ㄦ嫃閹间緡娲块柡鍌濐潐绾爼妫侀埀顒傛啺娓氬﹦?
        self.prev_model = None
        if self.model is not None and USE_DISTILL and task_idx > 0:
            print("Cloning current model as 'Old Model' for distillation...")
            self.prev_model = tf.keras.models.clone_model(self.model)
            self.prev_model.set_weights(self.model.get_weights())

        # 5. 濡絾鐗旈幑銏ゅ礉閳ュ磭绱︽俊?
        if self.model is None:
            print(f"\n>>> Building Initial Model...")
            self.model = self.model_builder_fn(self.global_input_spec)

        # 6. Representative replay
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

        # 7. 闁轰胶澧楀畵渚€宕濋悩鐑樼グ
        handler = FlowTransformer(self.pre_processor, **self.ft_components)
        handler.model_input_spec = self.global_input_spec
        handler.load_dataset(
            f"train_{task_idx}",
            combined_train_data,
            spec,
            self.class_manager.class_to_idx,
            evaluation_percent=0.0,
            fit_preprocessor=False
        )

        # 8. Training
        steps_per_epoch = max(1, int(handler.training_mask.sum()) // INCREMENTAL_BATCH_SIZE)
        print(f"\n>>> Training Task {task_idx} with method [{METHOD_NAME}] ...")
        print(f"[Train] Samples={int(handler.training_mask.sum())}, Batch={INCREMENTAL_BATCH_SIZE}, Steps/Epoch={steps_per_epoch}")
        self.model = handler.train_incremental(
            self.model,
            old_model=self.prev_model if USE_DISTILL else None,
            epochs=INCREMENTAL_EPOCHS,
            batch_size=INCREMENTAL_BATCH_SIZE,
            steps_per_epoch=steps_per_epoch,
            benign_class_idx=0,
            distill_lambda=DISTILL_LAMBDA if (USE_DISTILL and task_idx > 0) else 0.0,
            old_class_ids=old_class_ids if USE_DISTILL else None,
            supcon_lambda=SUPCON_LAMBDA if USE_SUPCON else 0.0,
            temperature=SUPCON_TEMP,
        )

        handler.clear_memory()
        del handler
        gc.collect()

        # 9. 闁哄洤鐡ㄩ弻?benign profile
        if UPDATE_BENIGN_PROFILE:
            benign_pool = raw_task_data[raw_task_data[spec.class_column] == 'Benign']
            if len(benign_pool) > 50000:
                benign_pool = benign_pool.sample(n=50000, random_state=42)

            if not benign_pool.empty:
                self.update_benign_profile(benign_pool, spec, is_first_task=(task_idx == 0))
            print("[Benign Profile] Updated.")
        else:
            print("[Benign Profile] Skipped.")

        # 10. 闁哄洤鐡ㄩ弻?memory
        if UPDATE_MEMORY:
            print(f"\n[Memory] Updating Rehearsal Memory...")
            feature_extractor_fn = lambda df: self._extract_features(df, spec)
            current_task_y_idx_raw = training_data_to_use[spec.class_column].map(self.class_manager.class_to_idx)
            self.memory.add(training_data_to_use, current_task_y_idx_raw, self.class_manager, feature_extractor_fn)
        else:
            print("[Memory] Skipped.")

        self.task_data_info.append(task_info)


    # def train_on_task(self, task_idx, task_info):
    #     raw_task_data = task_info['data']
    #     spec = task_info['spec']
    #     new_labels = task_info['new_labels']
    #
    #     self.class_manager.add_classes(new_labels)
    #     current_new_ids = [self.class_manager.class_to_idx[l] for l in new_labels]
    #     all_known_ids = list(self.class_manager.class_to_idx.values())
    #     old_class_ids = [i for i in all_known_ids if i not in current_new_ids]
    #
    #     training_data_to_use = raw_task_data
    #
    #     if task_idx == 0:
    #         print("Fitting Preprocessor on Task 0...")
    #         num_cols = set(spec.include_fields) - set(spec.categorical_fields)
    #         for c in num_cols:
    #             self.pre_processor.fit_numerical(c, pd.to_numeric(raw_task_data[c], errors='coerce').fillna(0).values)
    #         for c in spec.categorical_fields:
    #             if c != spec.class_column:
    #                 self.pre_processor.fit_categorical(c, raw_task_data[c].astype(str).values)
    #         self.update_input_spec(raw_task_data, spec)
    #
    #     if task_idx > 0:
    #         print(f"\n[Drift Logic] Checking Benign samples in Task {task_idx}...")
    #         df_benign = raw_task_data[raw_task_data[spec.class_column] == 'Benign']
    #         df_attack = raw_task_data[raw_task_data[spec.class_column] != 'Benign']
    #
    #         df_benign_filtered = self.filter_benign_via_drift(df_benign, spec)
    #         training_data_to_use = pd.concat([df_attack, df_benign_filtered], ignore_index=True)
    #
    #     if self.model is not None:
    #         print("Cloning current model as 'Old Model' for distillation...")
    #         self.prev_model = tf.keras.models.clone_model(self.model)
    #         self.prev_model.set_weights(self.model.get_weights())
    #
    #     if self.model is None:
    #         print(f"\n>>> Building Initial Model...")
    #         self.model = self.model_builder_fn(self.global_input_spec)
    #
    #     mem_data, mem_y_idx = self.memory.get()
    #     if not mem_data.empty:
    #         mem_data_rec = mem_data.copy()
    #         mem_data_rec[spec.class_column] = mem_y_idx.map(self.class_manager.idx_to_class)
    #         combined_train_data = pd.concat([training_data_to_use, mem_data_rec], ignore_index=True)
    #         print(f"[Train] Data: {len(training_data_to_use)} (New Data) + {len(mem_data_rec)} (Memory)")
    #     else:
    #         combined_train_data = training_data_to_use
    #         print(f"[Train] Data: {len(training_data_to_use)} (New Only)")
    #
    #     combined_train_data = combined_train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    #
    #     handler = FlowTransformer(self.pre_processor, **self.ft_components)
    #     handler.model_input_spec = self.global_input_spec
    #     handler.load_dataset(f"train_{task_idx}", combined_train_data, spec, self.class_manager.class_to_idx,
    #                          evaluation_percent=0.0, fit_preprocessor=False)
    #
    #     print(f"\n>>> Training Task {task_idx} (BCE + SupCon + Selective Distill)...")
    #     self.model = handler.train_incremental(
    #         self.model,
    #         old_model=self.prev_model,
    #         epochs=INCREMENTAL_EPOCHS,
    #         batch_size=INCREMENTAL_BATCH_SIZE,
    #         steps_per_epoch=256,
    #         benign_class_idx=0,
    #         distill_lambda=DISTILL_LAMBDA if task_idx > 0 else 0.0,
    #         old_class_ids=old_class_ids,
    #         supcon_lambda=SUPCON_LAMBDA,
    #         temperature=SUPCON_TEMP
    #     )
    #
    #     handler.clear_memory()
    #     del handler
    #     gc.collect()
    #
    #     benign_pool = raw_task_data[raw_task_data[spec.class_column] == 'Benign']
    #
    #     # 濞戞捁妗ㄧ花锟犳⒓閸欏?IDS2018 闁轰胶澧楀畵渚€鏌岃箛鎾变喊濠㈠爢鍐惧殼闁奸顥愰鍝ョ不濡ゅ嫮绠栭柟渚垮灪?OOM闁挎稑濂旂粭鍌炴⒔閹版澘娅氶柡?50,000 ?
    #     if len(benign_pool) > 50000:
    #         benign_pool = benign_pool.sample(n=50000, random_state=42)
    #
    #     if not benign_pool.empty:
    #         # self.update_benign_profile(benign_pool, spec, is_first_task=0.8 if task_idx > 0 else 0.0)
    #         self.update_benign_profile(benign_pool, spec, is_first_task=(task_idx == 0))
    #
    #     print(f"\n[Memory] Updating Rehearsal Memory using Representative Sampling (Herding)...")
    #     feature_extractor_fn = lambda df: self._extract_features(df, spec)
    #     current_task_y_idx_raw = training_data_to_use[spec.class_column].map(self.class_manager.class_to_idx)
    #     self.memory.add(training_data_to_use, current_task_y_idx_raw, self.class_manager, feature_extractor_fn)
    #
    #     self.task_data_info.append(task_info)

        # if task_idx == 0:
        #     benign_pool = raw_task_data[raw_task_data[spec.class_column] == 'Benign']
        #     self.update_benign_profile(benign_pool, spec, is_first_task=True)
        # else:
        #     self.update_benign_profile(df_benign_filtered, spec, is_first_task=False)
        #
        # print(f"\n[Memory] Updating Rehearsal Memory using Representative Sampling (Herding)...")
        # feature_extractor_fn = lambda df: self._extract_features(df, spec)
        # current_task_y_idx_raw = training_data_to_use[spec.class_column].map(self.class_manager.class_to_idx)
        # self.memory.add(training_data_to_use, current_task_y_idx_raw, self.class_manager, feature_extractor_fn)
        #
        # self.task_data_info.append(task_info)

    def update_input_spec(self, data, spec):
        num = [c for c in (set(spec.include_fields) - set(spec.categorical_fields)) if c in data.columns]
        cat = [c for c in spec.categorical_fields if c != spec.class_column and c in data.columns]
        levels = [len(self.pre_processor.encoded_levels.get(c, [])) for c in cat]
        self.global_input_spec = ModelInputSpecification(
            num + cat, len(num), levels, self.ft_components['input_encoding'].required_input_format
        )

    def evaluate_comprehensive(self, current_task_idx):
        print(f"\n>>> Comprehensive Evaluation for Task {current_task_idx} <<<")
        test_pool = self.task_data_info[0]['test_pool']
        full_test = self.task_data_info[0]['full_test']
        spec = self.task_data_info[0]['spec']
        current_benign_test = self.task_data_info[current_task_idx]['task_benign_test']
        current_new_attacks = self.task_data_info[current_task_idx]['new_labels']
        old_attacks = []
        if current_task_idx > 0:
            for i in range(current_task_idx):
                old_attacks.extend(self.task_data_info[i]['new_labels'])
        seen_attacks = old_attacks + current_new_attacks
        global_attacks = [str(lbl) for lbl in test_pool.keys() if str(lbl) != 'Benign']

        def build_eval_df(attack_labels, benign_df, mode_name):
            if mode_name == 'Global':
                return full_test.copy().reset_index(drop=True)
            if not attack_labels: return pd.DataFrame()
            dfs = []
            if benign_df is not None and not benign_df.empty:
                dfs.append(benign_df.copy().reset_index(drop=True))
            for label in attack_labels:
                if label in test_pool and not test_pool[label].empty: dfs.append(test_pool[label])
            if not dfs: return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True).reset_index(drop=True)

        def print_eval_build_info(df_eval, group_name, requested_attacks):
            label_counts = df_eval[spec.class_column].value_counts(dropna=False).sort_index()
            benign_count = int(label_counts.get('Benign', 0))
            attack_count = int(len(df_eval) - benign_count)
            label_counts_dict = {str(k): int(v) for k, v in label_counts.items()}
            present_attacks = [str(lbl) for lbl in label_counts.index if str(lbl) != 'Benign']
            missing_attacks = [lbl for lbl in requested_attacks if lbl not in present_attacks]

            print(
                f"[Eval Build] {group_name}: total={len(df_eval)}, "
                f"benign={benign_count}, attack={attack_count}, labels={label_counts_dict}"
            )
            print(
                f"[Eval Build] {group_name}: requested_attacks={requested_attacks}, "
                f"present_attacks={present_attacks}, missing_attacks={missing_attacks}"
            )

        def build_eval_class_to_idx(group_name, attack_labels):
            if group_name != 'Global':
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
            ('Global', global_attacks, None),
            ('TaskAware', seen_attacks, current_benign_test),
            ('New', current_new_attacks, current_benign_test),
            ('Old', old_attacks, current_benign_test)
        ]

        for group_name, attacks, benign_df in metrics_groups:
            if group_name == 'Old' and current_task_idx == 0: continue
            df_eval = build_eval_df(attacks, benign_df, group_name)
            if df_eval.empty: continue
            print_eval_build_info(df_eval, group_name, attacks)
            eval_class_to_idx = build_eval_class_to_idx(group_name, attacks)

            handler = FlowTransformer(self.pre_processor, **self.ft_components)
            handler.m = self.model
            handler.model_input_spec = self.global_input_spec

            res = handler.evaluate_binary(
                f"Eval_{group_name}_Task_{current_task_idx}", df_eval, spec,
                eval_class_to_idx, benign_class_idx=0, do_resampled_eval=True
            )
            for k, v in res.items():
                if isinstance(v, (int, float, np.floating)):
                    metrics_summary[f"{k}_{group_name}"] = float(v)
                elif isinstance(v, np.ndarray):
                    metrics_summary[f"{k}_{group_name}"] = v
            metrics_summary[f"requested_attacks_{group_name}"] = ",".join(attacks) if attacks else ""

            handler.clear_memory()
            del handler, df_eval
            gc.collect()
        return metrics_summary


# [main_fitday1.py] - incremental experiment entry

if __name__ == "__main__":
    ft_components = {
        'input_encoding': RecordLevelEmbed(64),
        'sequential_model': BasicTransformer(2, 128, n_heads=2),
        'classification_head': LastTokenClassificationHead(),
        'params': FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1),
    }

    dataset_path = r"/home/test/NF-UNSW-NB15-v3.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        exit()

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

        # --- 闁哄秶顭堢缓鐐┍椤旇姤鏆柨娑欒壘濠€顏嗘媼椤撶姷鐭婇柛鎾崇Х缁绘?Zero-Day 婵☆偀鍋撴繛?(Pre-Evaluation) ---
        zero_day_recall = 0.0
        if i > 0:
            # 濞达綀娉曢弫銈堛亹閹惧啿顤呮俊顖椻偓宕団偓鐑芥晬閸︾浛sk 0...i-1闁挎稑顦伴ˉ鍛?Task i 闁汇劌瀚弻濠囧绩鐠囨彃?
            zero_day_recall = learner.evaluate_zero_day(i, task)

        # 閻犱緡鍘剧划?
        learner.train_on_task(i, task)

        # 閻犱緡鍘剧划宀勫触鎼淬劎宕ｉ悹?
        print(f"\n>>> Evaluating Task {i} Performance (Comprehensive)...")
        res = learner.evaluate_comprehensive(i)

        record = {
            "Task": i,
            "Name": task['name'],
            "ZeroDay_Recall": zero_day_recall,
            "Acc_Global": res.get('real_acc_Global', np.nan),
            "BalAcc_Global": res.get('real_bal_acc_Global', np.nan),
            "F1_Global": res.get('real_f1_Global', np.nan),
            "Prec_Global": res.get('real_prec_Global', np.nan),
            "Rc_Global": res.get('real_recall_Global', np.nan),
            "FPR_Global": res.get('real_fpr_Global', np.nan),
            "PR_AUC_Global": res.get('real_pr_auc_Global', np.nan),
            "CM_Global": np.array2string(res.get('real_cm_Global', np.zeros((2, 2), dtype=int))),
            "Eval_Global_Status": res.get('eval_status_Global', 'missing'),
            "Eval_Global_Benign": res.get('n_benign_Global', 0),
            "Eval_Global_Attack": res.get('n_attack_Global', 0),
            "Acc_TaskAware": res.get('real_acc_TaskAware', np.nan),
            "BalAcc_TaskAware": res.get('real_bal_acc_TaskAware', np.nan),
            "F1_TaskAware": res.get('real_f1_TaskAware', np.nan),
            "Prec_TaskAware": res.get('real_prec_TaskAware', np.nan),
            "Rc_TaskAware": res.get('real_recall_TaskAware', np.nan),
            "FPR_TaskAware": res.get('real_fpr_TaskAware', np.nan),
            "PR_AUC_TaskAware": res.get('real_pr_auc_TaskAware', np.nan),
            "CM_TaskAware": np.array2string(res.get('real_cm_TaskAware', np.zeros((2, 2), dtype=int))),
            "Eval_TaskAware_Status": res.get('eval_status_TaskAware', 'missing'),
            "Eval_TaskAware_Benign": res.get('n_benign_TaskAware', 0),
            "Eval_TaskAware_Attack": res.get('n_attack_TaskAware', 0),
            "Rc_New": res.get('real_recall_New', np.nan),
            "Rc_Old": res.get('real_recall_Old', np.nan),
        }
        history.append(record)

        print(
            f"Result Task {i} [Global]: Acc={record['Acc_Global']:.4f}, BalAcc={record['BalAcc_Global']:.4f}, F1={record['F1_Global']:.4f}, "
            f"Prec={record['Prec_Global']:.4f}, Recall={record['Rc_Global']:.4f}, "
            f"FPR={record['FPR_Global']:.4f}, PR-AUC={record['PR_AUC_Global']:.4f}, "
            f"EvalStatus={record['Eval_Global_Status']}, Benign={int(record['Eval_Global_Benign'])}, Attack={int(record['Eval_Global_Attack'])}"
        )
        print(
            f"Result Task {i} [TaskAware]: Acc={record['Acc_TaskAware']:.4f}, BalAcc={record['BalAcc_TaskAware']:.4f}, F1={record['F1_TaskAware']:.4f}, "
            f"Prec={record['Prec_TaskAware']:.4f}, Recall={record['Rc_TaskAware']:.4f}, "
            f"FPR={record['FPR_TaskAware']:.4f}, PR-AUC={record['PR_AUC_TaskAware']:.4f}, "
            f"ZeroDay_Rc={zero_day_recall:.4f}, Learned_New_Rc={record['Rc_New']:.4f}, "
            f"EvalStatus={record['Eval_TaskAware_Status']}, Benign={int(record['Eval_TaskAware_Benign'])}, Attack={int(record['Eval_TaskAware_Attack'])}"
        )
        print(f"CM_Global:\n{record['CM_Global']}")
        print(f"CM_TaskAware:\n{record['CM_TaskAware']}")

    print("\n" + "=" * 30 + " Final Results Summary " + "=" * 30)
    final_df = pd.DataFrame(history)

    cols_order = [
        'Task', 'Name',
        'ZeroDay_Recall',
        'Acc_Global', 'BalAcc_Global', 'F1_Global', 'Prec_Global', 'Rc_Global', 'FPR_Global', 'PR_AUC_Global',
        'Eval_Global_Status', 'Eval_Global_Benign', 'Eval_Global_Attack',
        'Acc_TaskAware', 'BalAcc_TaskAware', 'F1_TaskAware', 'Prec_TaskAware', 'Rc_TaskAware', 'FPR_TaskAware', 'PR_AUC_TaskAware',
        'Eval_TaskAware_Status', 'Eval_TaskAware_Benign', 'Eval_TaskAware_Attack',
        'Rc_New', 'Rc_Old'
    ]
    existing_cols = [c for c in cols_order if c in final_df.columns]
    print(tabulate(final_df[existing_cols], headers="keys", tablefmt="grid", floatfmt=".4f"))

    if 'CM_Global' in final_df.columns:
        print("\n[Confusion Matrices - Global]")
        for _, row in final_df.iterrows():
            print(f"Task {row['Task']} ({row['Name']}):\n{row['CM_Global']}")

    if 'CM_TaskAware' in final_df.columns:
        print("\n[Confusion Matrices - TaskAware]")
        for _, row in final_df.iterrows():
            print(f"Task {row['Task']} ({row['Name']}):\n{row['CM_TaskAware']}")





