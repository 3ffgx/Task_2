from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from config import DATA_PATH, RANDOM_STATE, RESULT_DIR, ROBUSTNESS_SEEDS, TARGET_COL
from modeling import (
    bootstrap_ci,
    build_gpr_pipeline,
    calc_metrics,
    extrapolate_future_features,
    loocv_gpr_model,
    modal_best_params,
    nested_loocv_tree_model,
    quick_sensitivity_analysis,
    residual_tests,
    robustness_test_tree,
    save_feature_importance,
    save_future_forecast,
)
from plotting import (
    plot_correlation_heatmap,
    plot_feature_importance_pareto,
    plot_future_forecast,
    plot_model_comparison,
    plot_residual_diagnostics,
    plot_vif_summary,
)
from preprocess import build_feature_set, corr_filter, load_dataset, summarize_preprocessing, vif_filter

warnings.filterwarnings("ignore")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    print(f"数据已读取，样本数={len(df)}，字段数={df.shape[1]}")

    target_col = TARGET_COL
    print(f"当前目标变量：{target_col}")

    target_dir = RESULT_DIR / f"预测结果_{target_col}"
    target_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y, year = build_feature_set(df, target_col)
    X_before = X_raw.copy()
    print(f"初始特征数：{X_raw.shape[1]}")

    X_corr, corr_dropped, corr_matrix = corr_filter(X_raw, threshold=0.95)
    print(f"相关性初筛删除变量：{corr_dropped if corr_dropped else '无'}")

    X_final, vif_df, vif_dropped = vif_filter(X_corr)
    print(f"VIF筛选删除变量：{vif_dropped if vif_dropped else '无'}")
    print(f"最终特征数：{X_final.shape[1]}")

    summarize_preprocessing(X_before, X_final, corr_dropped, vif_dropped).to_excel(target_dir / "00_预处理摘要.xlsx", index=False)
    vif_df.to_excel(target_dir / "00_VIF检验结果.xlsx", index=False)
    corr_matrix.to_excel(target_dir / "00_相关性矩阵.xlsx", index=True)

    plot_correlation_heatmap(corr_matrix, str(target_dir / "图0_相关性热力图.png"))
    plot_vif_summary(vif_df, str(target_dir / "图0_VIF检验图.png"))

    et_base = ExtraTreesRegressor(random_state=RANDOM_STATE)
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE)
    et_param_grid = {"n_estimators": [50, 100, 150], "max_depth": [3, 5, None], "min_samples_split": [2, 5]}
    rf_param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2", None],
    }

    print("\n开始 ET 嵌套LOOCV...")
    y_pred_et, et_best_params_list, et_importances = nested_loocv_tree_model(X_final, y, et_base, et_param_grid, model_name="ET")
    print("\n开始 RF 嵌套LOOCV...")
    y_pred_rf, rf_best_params_list, rf_importances = nested_loocv_tree_model(X_final, y, rf_base, rf_param_grid, model_name="RF")
    print("\n开始 GPR LOOCV...")
    y_pred_gpr = loocv_gpr_model(X_final, y)

    met_et, met_rf, met_gpr = calc_metrics(y, y_pred_et), calc_metrics(y, y_pred_rf), calc_metrics(y, y_pred_gpr)
    test_et, test_rf, test_gpr = residual_tests(y, y_pred_et), residual_tests(y, y_pred_rf), residual_tests(y, y_pred_gpr)
    ci_et, ci_rf, ci_gpr = bootstrap_ci(y, y_pred_et), bootstrap_ci(y, y_pred_rf), bootstrap_ci(y, y_pred_gpr)
    robust_et = robustness_test_tree(X_final, y, ROBUSTNESS_SEEDS, model_name="ET")
    robust_rf = robustness_test_tree(X_final, y, ROBUSTNESS_SEEDS, model_name="RF")

    final_et = ExtraTreesRegressor(random_state=RANDOM_STATE, **modal_best_params(et_best_params_list))
    final_rf = RandomForestRegressor(random_state=RANDOM_STATE, **modal_best_params(rf_best_params_list))
    final_gpr = build_gpr_pipeline()
    final_et.fit(X_final, y)
    final_rf.fit(X_final, y)
    final_gpr.fit(X_final, y)

    future_X, future_feature_method = extrapolate_future_features(X_final, year, periods=3, window=5)
    future_feature_method.to_excel(target_dir / "08_future_feature_scenario.xlsx", index=False)
    future_forecast_df = save_future_forecast(future_X, target_col, final_et, final_rf, final_gpr, str(target_dir))

    eval_df = pd.DataFrame(
        {
            "Metric": ["R2", "MAE", "RMSE", "MAPE", "R2_CI_L", "R2_CI_U", "MAE_CI_L", "MAE_CI_U", "RMSE_CI_L", "RMSE_CI_U", "DW", "LjungBox_p", "BP_p", "Shapiro_p", "SeedR2_mean", "SeedR2_std", "SeedMAE_mean", "SeedMAE_std"],
            "ET": [met_et["R2"], met_et["MAE"], met_et["RMSE"], met_et["MAPE"], ci_et["R2_CI"][0], ci_et["R2_CI"][1], ci_et["MAE_CI"][0], ci_et["MAE_CI"][1], ci_et["RMSE_CI"][0], ci_et["RMSE_CI"][1], test_et["DW"], test_et["LjungBox_p"], test_et["BP_p"], test_et["Shapiro_p"], robust_et["r2_mean"], robust_et["r2_std"], robust_et["mae_mean"], robust_et["mae_std"]],
            "RF": [met_rf["R2"], met_rf["MAE"], met_rf["RMSE"], met_rf["MAPE"], ci_rf["R2_CI"][0], ci_rf["R2_CI"][1], ci_rf["MAE_CI"][0], ci_rf["MAE_CI"][1], ci_rf["RMSE_CI"][0], ci_rf["RMSE_CI"][1], test_rf["DW"], test_rf["LjungBox_p"], test_rf["BP_p"], test_rf["Shapiro_p"], robust_rf["r2_mean"], robust_rf["r2_std"], robust_rf["mae_mean"], robust_rf["mae_std"]],
            "GPR": [met_gpr["R2"], met_gpr["MAE"], met_gpr["RMSE"], met_gpr["MAPE"], ci_gpr["R2_CI"][0], ci_gpr["R2_CI"][1], ci_gpr["MAE_CI"][0], ci_gpr["MAE_CI"][1], ci_gpr["RMSE_CI"][0], ci_gpr["RMSE_CI"][1], test_gpr["DW"], test_gpr["LjungBox_p"], test_gpr["BP_p"], test_gpr["Shapiro_p"], np.nan, np.nan, np.nan, np.nan],
        }
    )
    eval_df.to_excel(target_dir / "01_模型评估结果.xlsx", index=False)

    pred_df = pd.DataFrame({"Year": year, "Actual": y, "ET_Pred": y_pred_et, "RF_Pred": y_pred_rf, "GPR_Pred": y_pred_gpr, "ET_Residual": y - y_pred_et, "RF_Residual": y - y_pred_rf, "GPR_Residual": y - y_pred_gpr})
    pred_df.to_excel(target_dir / "02_预测结果对照表.xlsx", index=False)
    pd.DataFrame(et_best_params_list).to_excel(target_dir / "03_ET最优参数.xlsx", index=False)
    pd.DataFrame(rf_best_params_list).to_excel(target_dir / "03_RF最优参数.xlsx", index=False)

    et_imp_df = save_feature_importance(X_final.columns.tolist(), et_importances, str(target_dir / "04_ET特征重要度.xlsx"))
    rf_imp_df = save_feature_importance(X_final.columns.tolist(), rf_importances, str(target_dir / "04_RF特征重要度.xlsx"))

    robust_df = pd.DataFrame({"Seed": ROBUSTNESS_SEEDS, "ET_R2": robust_et["r2_list"], "ET_MAE": robust_et["mae_list"], "RF_R2": robust_rf["r2_list"], "RF_MAE": robust_rf["mae_list"]})
    robust_df.to_excel(target_dir / "05_稳健性检验结果.xlsx", index=False)

    plot_model_comparison(year, y, y_pred_et, y_pred_gpr, y_pred_rf, target_col, str(target_dir / "图1_模型预测结果比较.png"))
    plot_residual_diagnostics(y_pred_et, y - y_pred_et, test_et["Shapiro_p"], test_et["BP_p"], str(target_dir / "图2_ET残差诊断图.png"))
    plot_future_forecast(year, y, future_forecast_df, target_col, str(target_dir / "图6_未来三年预测趋势.png"))
    plot_feature_importance_pareto(et_imp_df, "Importance", "Feature", "ET 模型特征重要度 Pareto 图", str(target_dir / "图3_ET特征重要度.png"), "#4c78a8")

    plot_feature_importance_pareto(
        pd.DataFrame({"Feature": ["Seed" + str(x) for x in ROBUSTNESS_SEEDS], "Importance": robust_et["r2_list"]}),
        "Importance",
        "Feature",
        "ET 多随机种子稳健性",
        str(target_dir / "图4_ET稳健性检验.png"),
        "#f58518",
    )

    try:
        import shap

        explainer = shap.TreeExplainer(final_et)
        shap_values = explainer.shap_values(X_final)
        shap_array = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values)
        mean_abs_shap = np.abs(shap_array).mean(axis=0) if shap_array.ndim == 2 else np.abs(shap_array)
        shap_importance = pd.DataFrame({"Feature": X_final.columns, "MeanAbsSHAP": mean_abs_shap}).sort_values("MeanAbsSHAP", ascending=False)
        shap_importance.to_excel(target_dir / "06_ET_SHAP特征重要度.xlsx", index=False)
        plot_feature_importance_pareto(shap_importance, "MeanAbsSHAP", "Feature", "ET 模型 SHAP 全局贡献 Pareto 图", str(target_dir / "图5_ET_SHAP特征重要性.png"), "#54a24b")
        print("SHAP 图和结果已生成。")
    except Exception as e:
        print(f"SHAP 结果生成失败：{e}")

    print("\n开始简化敏感性分析...")
    quick_sensitivity_analysis(df, target_col, str(target_dir))

    print("\n" + "=" * 70)
    print(f"目标变量：{target_col}")
    print(f"最终特征：{X_final.columns.tolist()}")
    print(f"ET -> R²={met_et['R2']:.4f}, MAE={met_et['MAE']:.4f}, RMSE={met_et['RMSE']:.4f}, MAPE={met_et['MAPE']:.4f}")
    print(f"RF -> R²={met_rf['R2']:.4f}, MAE={met_rf['MAE']:.4f}, RMSE={met_rf['RMSE']:.4f}, MAPE={met_rf['MAPE']:.4f}")
    print(f"GPR -> R²={met_gpr['R2']:.4f}, MAE={met_gpr['MAE']:.4f}, RMSE={met_gpr['RMSE']:.4f}, MAPE={met_gpr['MAPE']:.4f}")
    print(f"结果目录：{target_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
