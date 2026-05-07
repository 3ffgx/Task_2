from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import LEAKAGE_RULES, OPTIONAL_REDUNDANT_FEATURES, RANDOM_STATE, VIF_THRESHOLD, READ_ENCODINGS


def load_dataset(data_path: str | pd.PathLike[str]) -> pd.DataFrame:
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(data_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 文件编码无法识别。")


def build_feature_set(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    drop_cols = ["年份", target_col]
    for col in LEAKAGE_RULES.get(target_col, []):
        if col in df.columns and col not in drop_cols:
            drop_cols.append(col)
    for col in OPTIONAL_REDUNDANT_FEATURES:
        if col in df.columns and col not in drop_cols:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols, errors="ignore").copy()
    y = df[target_col].to_numpy()
    year = df["年份"].to_numpy()

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"以下列不是数值型，需先处理：{non_numeric_cols}")
    return X, y, year


def corr_filter(X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_new = X.drop(columns=to_drop, errors="ignore")
    return X_new, to_drop, corr


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    x_const = sm.add_constant(X)
    vif_df = pd.DataFrame(
        {
            "Feature": x_const.columns,
            "VIF": [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])],
        }
    )
    return (
        vif_df[vif_df["Feature"] != "const"]
        .sort_values(["VIF", "Feature"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def vif_filter(X: pd.DataFrame, vif_threshold: float = VIF_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    X_work = X.copy()
    dropped = []
    while X_work.shape[1] >= 2:
        vif_df = compute_vif(X_work)
        if vif_df["VIF"].max() <= vif_threshold:
            return X_work, vif_df, dropped
        bad_feature = vif_df.loc[0, "Feature"]
        dropped.append(bad_feature)
        X_work = X_work.drop(columns=[bad_feature])
    return X_work, compute_vif(X_work), dropped


def summarize_preprocessing(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    corr_dropped: List[str],
    vif_dropped: List[str],
) -> pd.DataFrame:
    rows = [
        ["原始特征数", X_before.shape[1]],
        ["相关性筛掉特征数", len(corr_dropped)],
        ["VIF筛掉特征数", len(vif_dropped)],
        ["最终特征数", X_after.shape[1]],
        ["最终特征名称", ", ".join(X_after.columns.tolist())],
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])
