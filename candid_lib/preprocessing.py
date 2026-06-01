# #  FlowTransformer 2023 by liamdm / liam@riftcs.com
#
# import numpy as np
# import pandas as pd
#
# from candid_lib.base_preprocessing import BasePreProcessing
# from candid_lib.enumerations import CategoricalFormat
#
#
# class StandardPreProcessing(BasePreProcessing):
#     def __init__(self, n_categorical_levels: int, clip_numerical_values:bool=False):
#         super().__init__()
#         self.n_categorical_levels:int = n_categorical_levels
#         self.clip_numerical_values:bool = clip_numerical_values
#         self.min_range = {}
#         self.encoded_levels = {}
#
#     @property
#     def name(self) -> str:
#         return "Standard Preprocessing"
#
#     @property
#     def parameters(self) -> dict:
#         return {
#             "n_categorical_levels": self.n_categorical_levels,
#             "clip_numerical_values": self.clip_numerical_values
#         }
#
#     def fit_numerical(self, column_name: str, values: np.array):
#
#         v0 = np.min(values)
#         v1 = np.max(values)
#         r = v1 - v0
#
#         self.min_range[column_name] = (v0, r)
#
#     def transform_numerical(self, column_name: str, values: np.array):
#         col_min, col_range = self.min_range[column_name]
#
#         if col_range == 0:
#             return np.zeros_like(values, dtype="float32")
#
#         # center on zero
#         values -= col_min
#
#         # apply a logarithm
#         col_values = np.log(values + 1)
#
#         # scale max to 1
#         col_values *= 1. / np.log(col_range + 1)
#
#         if self.clip_numerical_values:
#             col_values = np.clip(col_values, 0., 1.)
#
#         return col_values
#
#     def fit_categorical(self, column_name: str, values: np.array):
#         levels, level_counts = np.unique(values, return_counts=True)
#         sorted_levels = list(sorted(zip(levels, level_counts), key=lambda x: x[1], reverse=True))
#         self.encoded_levels[column_name] = [s[0] for s in sorted_levels[:self.n_categorical_levels]]
#
#
#     def transform_categorical(self, column_name:str, values: np.array, expected_categorical_format: CategoricalFormat):
#         encoded_levels = self.encoded_levels[column_name]
#         print(f"Encoding the {len(encoded_levels)} levels for {column_name}")
#
#         result_values = np.ones(len(values), dtype="uint32")
#         for level_i, level in enumerate(encoded_levels):
#             level_mask = values == level
#
#             # we use +1 here, as 0 = previously unseen, and 1 to (n + 1) are the encoded levels
#             result_values[level_mask] = level_i + 1
#
#         if expected_categorical_format == CategoricalFormat.Integers:
#             return result_values
#
#         v = pd.get_dummies(result_values, prefix=column_name)
#         return v


#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import numpy as np
import pandas as pd

from candid_lib.base_preprocessing import BasePreProcessing
from candid_lib.enumerations import CategoricalFormat


class StandardPreProcessing(BasePreProcessing):
    def __init__(self, n_categorical_levels: int, clip_numerical_values: bool = False):
        super().__init__()
        self.n_categorical_levels: int = n_categorical_levels
        self.clip_numerical_values: bool = clip_numerical_values
        self.min_range = {}
        self.encoded_levels = {}

    @property
    def name(self) -> str:
        return "Standard Preprocessing"

    @property
    def parameters(self) -> dict:
        return {
            "n_categorical_levels": self.n_categorical_levels,
            "clip_numerical_values": self.clip_numerical_values
        }

    def fit_numerical(self, column_name: str, values: np.array):
        values = values[values >= 0]
        if values.size == 0:
            v0, v1, r = 0.0, 0.0, 0.0
        else:
            v0 = np.min(values)
            v1 = np.max(values)
            r = v1 - v0

        self.min_range[column_name] = (v0, r)

    def transform_numerical(self, column_name: str, values: np.array):
        if column_name not in self.min_range:
            print(f"[WARN] {column_name} not found in min_range, fitting on the fly. This might indicate an issue.")
            self.fit_numerical(column_name, values)

        col_min, col_range = self.min_range[column_name]

        if col_range == 0:
            return np.zeros_like(values, dtype="float32")

        # (!!!) (BUG 淇) (!!!)

        # 1. (淇) 瑁佸壀鍊?(Clip values):
        #    灏嗗湪娴嬭瘯闆嗕腑鍙兘鍑虹幇鐨勫皬浜?col_min 鐨勫€? 瑁佸壀鍒?col_min銆?        values_clipped = np.clip(values, col_min, None)

        # 2. (淇) 涓績鍖?(Center on zero):
        #    鐜板湪 values_clipped - col_min 鐨勬渶灏忓€兼槸 0銆?        values_centered = values_clipped - col_min

        # 3. (淇) 搴旂敤瀵规暟 (Apply a logarithm):
        #    鐜板湪鏄?np.log(0 + 1) 璧锋, 涓嶄細浜х敓 NaN銆?        col_values = np.log(values_centered + 1)

        # 4. (淇) 缂╂斁 (Scale max to 1):
        log_range = np.log(col_range + 1)
        if log_range == 0:
            # (閬垮厤闄や互闆?
            return np.zeros_like(values, dtype="float32")

        col_values *= 1. / log_range
        # (!!!) (淇缁撴潫) (!!!)

        if self.clip_numerical_values:
            # 瑁佸壀閭ｄ簺澶т簬璁粌闆嗘渶澶у€肩殑鍊?(鍙€?
            col_values = np.clip(col_values, 0., 1.)

        return col_values

    def fit_categorical(self, column_name: str, values: np.array):
        levels, level_counts = np.unique(values, return_counts=True)
        sorted_levels = list(sorted(zip(levels, level_counts), key=lambda x: x[1], reverse=True))
        self.encoded_levels[column_name] = [s[0] for s in sorted_levels[:self.n_categorical_levels]]

    def transform_categorical(self, column_name: str, values: np.array, expected_categorical_format: CategoricalFormat):
        if column_name not in self.encoded_levels:
            print(f"[WARN] {column_name} not fitted, fitting on the fly. This might indicate an issue.")
            self.fit_categorical(column_name, values)

        encoded_levels = self.encoded_levels[column_name]
        print(f"Encoding the {len(encoded_levels)} levels for {column_name}")

        result_values = np.zeros(len(values), dtype="uint32")  # (榛樿鍊间负 0 = 'unseen')
        for level_i, level in enumerate(encoded_levels):
            level_mask = values == level

            # we use +1 here, as 0 = previously unseen, and 1 to (n) are the encoded levels
            result_values[level_mask] = level_i + 1

        if expected_categorical_format == CategoricalFormat.Integers:
            return result_values

        v = pd.get_dummies(result_values, prefix=column_name)

        # (纭繚鎵€鏈夊彲鑳界殑鍒楅兘瀛樺湪, 鍗充娇鍦ㄨ繖涓壒娆′腑娌℃湁鍑虹幇)
        expected_cols = [f"{column_name}_{i + 1}" for i in range(len(encoded_levels))]
        # (娣诲姞 unseen '0' 鍒?
        expected_cols = [f"{column_name}_0"] + expected_cols

        v = v.reindex(columns=expected_cols, fill_value=0)

        return v




