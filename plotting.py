from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import VIF_THRESHOLD

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_model_comparison(year: np.ndarray, y_true: np.ndarray, y_et: np.ndarray, y_gpr: np.ndarray, y_rf: np.ndarray, target_col: str, out_path: str):
    plt.figure(figsize=(12, 6))
    plt.plot(year, y_true, "o-", label="Actual", linewidth=2)
    plt.plot(year, y_et, "s-", label="ET", linewidth=2)
    plt.plot(year, y_rf, "d-", label="RF", linewidth=2)
    plt.plot(year, y_gpr, "^-", label="GPR", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel(target_col)
    plt.title(f"{target_col} 不同模型预测结果比较")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual_diagnostics(y_pred: np.ndarray, residuals: np.ndarray, shapiro_p: float, bp_p: float, out_prefix: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    import scipy.stats as ss

    ss.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q图 (Shapiro p={shapiro_p:.3f})")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(y_pred, residuals, alpha=0.7, edgecolors="k")
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Fitted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"Residuals vs Fitted (BP p={bp_p:.3f})")
    plt.tight_layout()
    plt.savefig(out_prefix, dpi=300)
    plt.close()


def plot_future_forecast(
    year: np.ndarray,
    y: np.ndarray,
    future_df: pd.DataFrame,
    target_col: str,
    out_path: str,
):
    scale = 10000.0
    year = np.asarray(year, dtype=int)
    y_plot = np.asarray(y, dtype=float) / scale
    future_year = future_df["Year"].to_numpy(dtype=int)
    last_year = int(np.nanmax(year))
    last_value = float(y_plot[np.nanargmax(year)])

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(year, y_plot, color="#1f77b4", marker="o", markersize=4.8, linewidth=2.2, label="历史实际值")
    ax.axvspan(future_year.min() - 0.5, future_year.max() + 0.5, color="#f2f2f2", alpha=0.9)
    ax.axvline(last_year + 0.5, color="#777777", linestyle="--", linewidth=1.2)

    forecast_styles = [
        ("ET_Forecast", "ET主预测", "#d62728", 2.4),
        ("RF_Forecast", "RF对照预测", "#2ca02c", 1.8),
        ("GPR_Forecast", "GPR对照预测", "#9467bd", 1.8),
        ("Mean_Forecast", "三模型均值", "#ff7f0e", 2.0),
    ]
    anchor_year = np.r_[last_year, future_year]
    for col, label, color, linewidth in forecast_styles:
        values = np.r_[last_value, future_df[col].to_numpy(dtype=float) / scale]
        ax.plot(anchor_year, values, color=color, marker="o", markersize=4.6, linewidth=linewidth, linestyle="--", label=label)

    min_forecast = future_df[["ET_Forecast", "RF_Forecast", "GPR_Forecast"]].min(axis=1).to_numpy(dtype=float) / scale
    max_forecast = future_df[["ET_Forecast", "RF_Forecast", "GPR_Forecast"]].max(axis=1).to_numpy(dtype=float) / scale
    ax.fill_between(future_year, min_forecast, max_forecast, color="#d62728", alpha=0.08, label="模型预测范围")

    ymin, ymax = ax.get_ylim()
    ax.text(future_year.mean(), ymax - (ymax - ymin) * 0.05, "预测期", ha="center", va="top", fontsize=10, color="#555555")
    ax.set_title(f"{target_col}历史变化及未来三年预测", fontsize=14, pad=12)
    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel(f"{target_col}（万hm2）", fontsize=12)
    ax.set_xlim(year.min() - 0.5, future_year.max() + 0.6)
    ax.set_xticks(np.arange(year.min(), future_year.max() + 1, 2))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=9, ncol=3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, out_path: str):
    values = corr_matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    fig_w = max(8.5, len(corr_matrix.columns) * 0.52)
    fig_h = max(7.0, len(corr_matrix.index) * 0.46)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#e8e8e8")
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_matrix.index, fontsize=9)
    ax.set_title("特征相关性热力图", fontsize=13, pad=12)
    ax.set_xticks(np.arange(-0.5, len(corr_matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(corr_matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    if len(corr_matrix.columns) <= 10:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black" if abs(values[i, j]) < 0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vif_summary(vif_df: pd.DataFrame, out_path: str, threshold: float = VIF_THRESHOLD):
    vif_plot = vif_df.sort_values("VIF", ascending=True).reset_index(drop=True)
    fig_h = max(4.2, len(vif_plot) * 0.55)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    colors = ["#d55e00" if v > threshold else "#4c78a8" for v in vif_plot["VIF"]]
    ax.barh(vif_plot["Feature"], vif_plot["VIF"], color=colors, edgecolor="white")
    ax.axvline(threshold, color="#b22222", linestyle="--", linewidth=1.6, label=f"阈值={threshold:g}")
    for y_pos, val in enumerate(vif_plot["VIF"]):
        ax.text(val + 0.06, y_pos, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("VIF")
    ax.set_ylabel("特征")
    ax.set_title("多重共线性检验结果", fontsize=13, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_pareto(df: pd.DataFrame, value_col: str, label_col: str, title: str, out_path: str, color: str):
    data = df.sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    total = float(data[value_col].sum())
    data["CumPct"] = 0.0 if total == 0 else data[value_col].cumsum() / total * 100
    fig, ax1 = plt.subplots(figsize=(9.2, max(5.4, len(data) * 0.65)))
    x = np.arange(len(data))
    ax1.bar(x, data[value_col], color=color, width=0.65, alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(data[label_col], rotation=30, ha="right")
    ax1.set_ylabel(value_col)
    ax1.set_title(title, fontsize=13, pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, data["CumPct"], color="#b22222", marker="o", linewidth=1.8)
    ax2.set_ylabel("累计贡献率 (%)")
    ax2.set_ylim(0, 105)
    for xi, val, pct in zip(x, data[value_col], data["CumPct"]):
        ax1.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.text(xi, pct + 1.2, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, color="#7a1f1f")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
