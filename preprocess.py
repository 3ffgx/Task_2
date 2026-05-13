from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import LEAKAGE_RULES, VIF_THRESHOLD, READ_ENCODINGS


# 原始表头中可能带有单位、换行或空格，这里统一转换为代码内部使用的标准列名。
COLUMN_ALIASES = {
    "常驻人口 / 万人": "常驻人口",
    "区域面积 A (km2)": "区域面积",
    "平均年降水量 P (mm)": "平均年降水量",
    "降水量换算 (亿 m3)": "降水量换算",
    "水资源总量 W (亿 m3)": "水资源总量",
    "产水系数 α": "产水系数",
    "产水模数 M (m3/ hm2)": "产水模数",
    "生产用水 / 亿 m3": "生产用水",
    "生活用水 / 亿 m3": "生活用水",
    "生态用水 / 亿 m3": "生态用水",
    "人均生产用水 /m3": "人均生产用水",
    "人均生活用水 /m3": "人均生活用水",
    "人均生态用水 /m3": "人均生态用水",
    "总水资源生态足迹EFW(hm2)": "总水资源生态足迹",
    "总水资源生态承载力 ECw": "总水资源生态承载力",
    "人均水资源生态足迹EFW(hm2/ 人)": "人均水资源生态足迹",
    "人均水资源承载力 ECw (hm2/ 人)": "人均水资源承载力",
    "万元 GDP 水生态足迹EFGDP hm2/ 万元": "万元GDP水生态足迹",
    "水资源生态盈余 赤字(hm2)": "水资源生态盈余赤字",
}


def normalize_column_name(column: object) -> str:
    """清理单个表头，去除多余空白并匹配标准列名。"""
    name = " ".join(str(column).replace("\n", " ").split())
    return COLUMN_ALIASES.get(name, name)


def load_dataset(data_path: str | pd.PathLike[str]) -> pd.DataFrame:
    """读取 CSV 数据，并自动尝试多种编码。"""
    for enc in READ_ENCODINGS:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            df = df.dropna(how="all").copy()
            df.columns = [normalize_column_name(col) for col in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 文件编码无法识别。")


def build_feature_set(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """构建建模输入 X、目标变量 y 和年份序列。"""
    # 年份、目标变量和手动控制剔除变量不进入模型。
    drop_cols = ["年份", target_col]
    for col in LEAKAGE_RULES.get(target_col, []):
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
    """基于两两相关系数进行初筛，剔除相关系数超过阈值的冗余特征。"""
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_new = X.drop(columns=to_drop, errors="ignore")
    return X_new, to_drop, corr


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """计算各特征的方差膨胀因子，用于诊断多重共线性。"""
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
    """迭代剔除 VIF 最大且超过阈值的特征，直到所有特征 VIF 达标。"""
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
    """整理预处理摘要，便于导出 Excel 作为建模记录。"""
    rows = [
        ["剩余待筛选特征数", X_before.shape[1]],
        ["相关性筛掉特征数", len(corr_dropped)],
        ["VIF筛掉特征数", len(vif_dropped)],
        ["最终特征数", X_after.shape[1]],
        ["最终特征名称", ", ".join(X_after.columns.tolist())],
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])
