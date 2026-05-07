from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from config import BOOTSTRAP_ITER, RANDOM_STATE, ROBUSTNESS_SEEDS, VIF_THRESHOLD
from preprocess import corr_filter, vif_filter


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }


def residual_tests(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    res = y_true - y_pred
    n = len(res)
    out = {"DW": np.nan, "LjungBox_p": np.nan, "BP_p": np.nan, "Shapiro_p": np.nan}
    out["DW"] = durbin_watson(res)
    try:
        out["LjungBox_p"] = acorr_ljungbox(res, lags=min(5, max(1, n // 2)), return_df=True)["lb_pvalue"].iloc[-1]
    except Exception:
        pass
    try:
        out["BP_p"] = het_breuschpagan(res, sm.add_constant(y_pred.reshape(-1, 1)))[1]
    except Exception:
        pass
    try:
        if n <= 5000:
            out["Shapiro_p"] = stats.shapiro(res)[1]
        else:
            out["Shapiro_p"] = stats.kstest(res, "norm", args=(np.mean(res), np.std(res)))[1]
    except Exception:
        pass
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_iter: int = BOOTSTRAP_ITER, seed: int = RANDOM_STATE) -> Dict[str, Tuple[float, float]]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    r2_list, mae_list, rmse_list = [], [], []
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        r2_list.append(r2_score(yt, yp))
        mae_list.append(mean_absolute_error(yt, yp))
        rmse_list.append(np.sqrt(mean_squared_error(yt, yp)))
    return {
        "R2_CI": (np.percentile(r2_list, 2.5), np.percentile(r2_list, 97.5)),
        "MAE_CI": (np.percentile(mae_list, 2.5), np.percentile(mae_list, 97.5)),
        "RMSE_CI": (np.percentile(rmse_list, 2.5), np.percentile(rmse_list, 97.5)),
    }


def nested_loocv_tree_model(
    X: pd.DataFrame,
    y: np.ndarray,
    base_model,
    param_grid: Dict,
    model_name: str,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    best_params_list: List[Dict] = []
    importances = np.zeros(X.shape[1], dtype=float)

    for i, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        inner_cv = KFold(n_splits=min(3, len(y_train)), shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=clone(base_model),
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred[test_idx] = best_model.predict(X_test)
        best_params_list.append(grid.best_params_)
        if hasattr(best_model, "feature_importances_"):
            importances += best_model.feature_importances_
        if i % 5 == 0 or i == len(y):
            print(f"[{model_name}] 外层LOOCV进度：{i}/{len(y)}")

    importances /= len(y)
    return y_pred, best_params_list, importances


def loocv_gpr_model(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    model = build_gpr_pipeline()
    for i, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
        if i % 5 == 0 or i == len(y):
            print(f"[GPR] LOOCV进度：{i}/{len(y)}")
    return y_pred


def robustness_test_tree(X: pd.DataFrame, y: np.ndarray, seeds: List[int], model_name: str = "ET") -> Dict[str, float]:
    loo = LeaveOneOut()
    r2_list, mae_list = [], []
    for seed in seeds:
        y_pred = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y[train_idx]
            if model_name == "ET":
                model = ExtraTreesRegressor(n_estimators=100, max_depth=3, min_samples_split=2, random_state=seed)
            elif model_name == "RF":
                model = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_split=2, random_state=seed)
            else:
                raise ValueError("model_name 仅支持 ET 或 RF")
            model.fit(X_train, y_train)
            y_pred[test_idx] = model.predict(X_test)
        r2_list.append(r2_score(y, y_pred))
        mae_list.append(mean_absolute_error(y, y_pred))
    return {
        "r2_mean": np.mean(r2_list),
        "r2_std": np.std(r2_list),
        "mae_mean": np.mean(mae_list),
        "mae_std": np.std(mae_list),
        "r2_list": r2_list,
        "mae_list": mae_list,
    }


def modal_best_params(best_params_list: List[Dict]) -> Dict:
    normalized = [tuple(sorted(d.items())) for d in best_params_list]
    mode_tuple, _ = Counter(normalized).most_common(1)[0]
    return dict(mode_tuple)


def build_gpr_pipeline() -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def extrapolate_future_features(
    X: pd.DataFrame,
    year: np.ndarray,
    periods: int = 3,
    window: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years = np.asarray(year, dtype=float)
    future_years = np.arange(int(np.nanmax(years)) + 1, int(np.nanmax(years)) + periods + 1)
    future_X = pd.DataFrame(index=future_years, columns=X.columns, dtype=float)
    method_rows = []

    for col in X.columns:
        values = pd.Series(X[col].to_numpy(dtype=float), index=years).dropna()
        if values.empty:
            future_values = np.repeat(np.nan, periods)
            method = "missing"
        elif len(values) == 1:
            future_values = np.repeat(values.iloc[-1], periods)
            method = "last_observation"
        else:
            recent = values.tail(min(window, len(values)))
            x_recent = recent.index.to_numpy(dtype=float)
            y_recent = recent.to_numpy(dtype=float)
            x0 = x_recent[0]
            slope, intercept = np.polyfit(x_recent - x0, y_recent, deg=1)
            raw_pred = intercept + slope * (future_years - x0)
            lower = np.nanpercentile(values.to_numpy(dtype=float), 5)
            upper = np.nanpercentile(values.to_numpy(dtype=float), 95)
            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                future_values = np.clip(raw_pred, lower, upper)
                method = f"linear_trend_last_{len(recent)}yr_clipped_p5_p95"
            else:
                future_values = raw_pred
                method = f"linear_trend_last_{len(recent)}yr"

        future_X[col] = future_values
        for fy, fv in zip(future_years, future_values):
            method_rows.append({"Year": int(fy), "Feature": col, "Forecast_Value": fv, "Method": method})

    future_X.insert(0, "Year", future_years.astype(int))
    return future_X, pd.DataFrame(method_rows)


def save_future_forecast(
    future_X_with_year: pd.DataFrame,
    target_col: str,
    final_et,
    final_rf,
    final_gpr,
    out_dir: str,
) -> pd.DataFrame:
    feature_cols = [c for c in future_X_with_year.columns if c != "Year"]
    future_X = future_X_with_year[feature_cols]
    out = pd.DataFrame(
        {
            "Year": future_X_with_year["Year"].astype(int),
            "Scenario": "Recent trend scenario",
            "Target": target_col,
            "ET_Forecast": final_et.predict(future_X),
            "RF_Forecast": final_rf.predict(future_X),
            "GPR_Forecast": final_gpr.predict(future_X),
        }
    )
    out["Mean_Forecast"] = out[["ET_Forecast", "RF_Forecast", "GPR_Forecast"]].mean(axis=1)
    out["Main_Forecast_ET"] = out["ET_Forecast"]
    for col in feature_cols:
        out[f"X_{col}"] = future_X[col].to_numpy()
    out.to_excel(os.path.join(out_dir, "08_future_3yr_forecast.xlsx"), index=False)
    return out


def save_feature_importance(feature_names: List[str], importances: np.ndarray, out_path: str) -> pd.DataFrame:
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
    df_imp["Pct"] = (df_imp["Importance"] / df_imp["Importance"].sum() * 100).round(2)
    df_imp.to_excel(out_path, index=False)
    return df_imp


def quick_sensitivity_analysis(df: pd.DataFrame, target_col: str, out_dir: str) -> pd.DataFrame:
    results = []

    def _find_col(data: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in data.columns:
                return c
        return None

    def _real_columns(cols: List[str], data: pd.DataFrame) -> List[str]:
        out = []
        for col in cols:
            if col in data.columns and col not in out:
                out.append(col)
        return out

    year_col = _find_col(df, ["年份", "Year", "year"])
    life_col = _find_col(df, ["生活用水", "生活用水量", "生活用水 / 亿 m³", "生活用水/亿m3"])
    eco_col = _find_col(df, ["生态用水", "生态环境用水", "生态环境用水量", "生态用水 / 亿 m³", "生态用水/亿m3"])
    agri_col = _find_col(df, ["农业用水", "农业用水量", "农业用水 / 亿 m³", "农业用水/亿m3", "第一产业用水"])
    ind_col = _find_col(df, ["工业用水量", "工业用水", "工业用水 / 亿 m³", "工业用水/亿m3", "第二产业用水"])
    prod_col = _find_col(df, ["生产用水", "生产用水量", "生产用水 / 亿 m³", "生产用水/亿m3"])
    total_col = _find_col(df, ["总用水量", "总用水", "用水总量", "总用水量 / 亿 m³", "总用水量/亿m3"])

    full_features = [
        "常驻人口", "常住人口",
        "平均年降水量", "降水量", "降水量换算",
        "水资源总量", "水资源总量 W", "水资源总量 W (亿 m³)",
        "生产用水", "生活用水",
    ]
    no_direct_water_features = [
        "常驻人口", "常住人口",
        "平均年降水量", "降水量", "降水量换算",
        "水资源总量", "水资源总量 W", "水资源总量 W (亿 m³)",
    ]

    def _calc_total_use(data: pd.DataFrame) -> pd.Series:
        parts = []
        if agri_col is not None and ind_col is not None:
            parts.extend([agri_col, ind_col])
        elif prod_col is not None:
            parts.append(prod_col)
        if life_col is not None:
            parts.append(life_col)
        if eco_col is not None:
            parts.append(eco_col)
        parts = [c for c in parts if c in data.columns]
        if parts:
            return data[parts].sum(axis=1)
        if total_col is not None and total_col in data.columns:
            return data[total_col]
        raise KeyError("无法计算总用水量。")

    def _detect_target_unit(data: pd.DataFrame) -> str:
        try:
            total_use = _calc_total_use(data.copy())
            ef_hm2 = 5.19 * total_use * 1e8 / 3140
            ef_wan_hm2 = ef_hm2 / 1e4
            y = data[target_col]
            err_hm2 = np.nanmedian(np.abs(y - ef_hm2))
            err_wan = np.nanmedian(np.abs(y - ef_wan_hm2))
            return "wan_hm2" if err_wan < err_hm2 else "hm2"
        except Exception:
            return "hm2"

    target_unit = _detect_target_unit(df)

    def _rebuild_eco_scenario(mode: str) -> Tuple[pd.DataFrame, float]:
        if year_col is None or life_col is None or eco_col is None:
            raise KeyError("缺少年份、生活用水或生态用水列。")
        data = df.copy()
        ref = data[data[year_col].between(2020, 2023)].copy()
        if ref.empty:
            raise ValueError("没有找到 2020—2023 年参考期数据。")
        ratio = ref[eco_col] / ref[life_col]
        if mode == "mean":
            r = ratio.mean()
        elif mode == "min":
            r = ratio.min()
        elif mode == "max":
            r = ratio.max()
        else:
            raise ValueError("mode 只能是 mean、min 或 max。")
        fill_mask = data[year_col].between(2011, 2019)
        data.loc[fill_mask, eco_col] = data.loc[fill_mask, life_col] * r
        if agri_col is not None and ind_col is not None:
            if prod_col is not None:
                data[prod_col] = data[agri_col] + data[ind_col]
            else:
                data["生产用水"] = data[agri_col] + data[ind_col]
        new_total_col = total_col if total_col is not None else "总用水量"
        data[new_total_col] = _calc_total_use(data)
        ef_hm2 = 5.19 * data[new_total_col] * 1e8 / 3140
        data[target_col] = ef_hm2 / 1e4 if target_unit == "wan_hm2" else ef_hm2
        return data, float(r)

    def run_one(df_s: pd.DataFrame, scenario_type: str, scenario_name: str, feature_cols: List[str], note: str = ""):
        feature_cols_real = _real_columns(feature_cols, df_s)
        if not feature_cols_real or target_col not in df_s.columns:
            return
        X = df_s[feature_cols_real].copy()
        y = df_s[target_col].to_numpy()
        corr_dropped, vif_dropped = [], []
        if X.shape[1] >= 2:
            X_corr, corr_dropped, _ = corr_filter(X, threshold=0.95)
            X_final = X_corr.copy()
            if X_corr.shape[1] >= 2:
                X_final, _, vif_dropped = vif_filter(X_corr, vif_threshold=VIF_THRESHOLD)
        else:
            X_final = X.copy()
        if X_final.shape[1] == 0:
            return
        et_base = ExtraTreesRegressor(random_state=RANDOM_STATE)
        et_param_grid = {"n_estimators": [100], "max_depth": [3, 5, None], "min_samples_split": [2]}
        y_pred, _, importances = nested_loocv_tree_model(X_final, y, et_base, et_param_grid, model_name=f"敏感性_{scenario_name}")
        metrics = calc_metrics(y, y_pred)
        imp_df = pd.DataFrame({"Feature": X_final.columns, "Importance": importances}).sort_values("Importance", ascending=False)
        results.append(
            {
                "敏感性类型": scenario_type,
                "情景": scenario_name,
                "说明": note,
                "样本数": len(df_s),
                "目标单位判断": target_unit,
                "初始特征": ", ".join(feature_cols_real),
                "最终特征": ", ".join(X_final.columns.tolist()),
                "相关性删除变量": ", ".join(corr_dropped) if corr_dropped else "无",
                "VIF删除变量": ", ".join(vif_dropped) if vif_dropped else "无",
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "前三重要变量": " > ".join(imp_df["Feature"].head(3).tolist()),
            }
        )

    for mode, name in [("mean", "S0_生态用水平均比值"), ("min", "S1_生态用水最小比值"), ("max", "S2_生态用水最大比值")]:
        try:
            df_s, ratio_used = _rebuild_eco_scenario(mode)
            run_one(df_s, "生态用水补充方式", name, full_features, note=f"使用2020—2023年生态用水/生活用水{mode}比值补充2011—2019年生态用水；比值={ratio_used:.6f}。")
        except Exception as e:
            results.append({"敏感性类型": "生态用水补充方式", "情景": name, "说明": "生态用水补充情景执行失败。", "错误信息": str(e)})

    run_one(
        df,
        "特征组合补充分析",
        "F1_去除直接用水变量",
        no_direct_water_features,
        note="用于检验模型脱离生产/生活/生态等直接用水变量后的表现。",
    )

    result_df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, "07_生态用水补充敏感性分析.xlsx")
    result_df.to_excel(out_path, index=False)
    print(f"生态用水补充敏感性分析完成：{out_path}")
    return result_df
