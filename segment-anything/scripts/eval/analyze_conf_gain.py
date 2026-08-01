#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""置信度—參考增益相關性分析。

論文第 4.4 節報告置信度的空間分布，第 4.5.2 節報告參考影像效益隨天候而異，
兩者原本只用一句「與置信度統計相符」連接——那是敘述，不是量測。本腳本把它
變成可檢驗的統計量：對齊置信度（conf_mean）是否真的能預測參考增益（gain）。

判準（動手前定死，不因結果調整）：
    (a) 整體與條件內皆顯著正相關（p < 0.05）→ 因果鏈成立。
    (b) 僅整體顯著、條件內不顯著            → 條件與置信度共變，無法分離。
    (c) 無顯著相關                          → 條件依賴另有來源，需修正 4.5.2 節敘述。

夜間影像的置信度天生偏低，若不控制條件，跨條件相關性可能只是在重述
「夜間表現差」，因此必須同時報告條件內（within-condition）的相關係數，
不能只看整體數字。

輸入：figures_defense/reference_gain/per_image.csv（Task 1 產出，406 列）
輸出：
    figures_defense/conf_gain_scatter.png   散佈圖 + 整體迴歸線 + 95% 信賴區間
    figures_defense/conf_gain_stats.json    相關係數、p 值、分箱結果

用法：
    conda run -n sam_env python scripts/eval/analyze_conf_gain.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # segment-anything/
CSV_PATH = os.path.join(ROOT, "figures_defense", "reference_gain", "per_image.csv")
OUT_DIR = os.path.join(ROOT, "figures_defense")
os.makedirs(OUT_DIR, exist_ok=True)

COND = ["fog", "rain", "snow", "night"]
COND_LABEL = {"fog": "Fog", "rain": "Rain", "snow": "Snow", "night": "Night"}
COND_COLOR = {"fog": "#3a7ea3", "rain": "#4a8a68", "snow": "#c07526", "night": "#b0473d"}
COND_MARKER = {"fog": "o", "rain": "s", "snow": "^", "night": "D"}
C_INK = "#16222a"
SIG_ALPHA = 0.05  # 顯著性門檻

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#b8c6cb",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e3eaed",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 160,
})


def correlate(df, tag):
    """計算 Pearson r、Spearman rho 及其 p 值（scipy.stats）。"""
    r, pr = stats.pearsonr(df["conf_mean"], df["gain"])
    rho, prho = stats.spearmanr(df["conf_mean"], df["gain"])
    return dict(
        scope=tag,
        n=int(len(df)),
        pearson_r=round(float(r), 4),
        pearson_p=round(float(pr), 5),
        spearman_rho=round(float(rho), 4),
        spearman_p=round(float(prho), 5),
        mean_gain=round(float(df["gain"].mean()), 4),
        significant=bool(pr < SIG_ALPHA),
    )


def ols_ci_band(x, y, n_points=200):
    """整體 OLS 迴歸線與 95% 信賴帶（mean-response confidence band，非預測帶）。"""
    res = stats.linregress(x, y)
    n = len(x)
    dof = n - 2
    x_grid = np.linspace(x.min(), x.max(), n_points)
    y_hat = res.intercept + res.slope * x_grid

    residuals = y - (res.intercept + res.slope * x)
    s_err = np.sqrt(np.sum(residuals ** 2) / dof)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    se_mean = s_err * np.sqrt(1.0 / n + (x_grid - x_mean) ** 2 / sxx)
    t_val = stats.t.ppf(0.975, dof)
    ci = t_val * se_mean
    return x_grid, y_hat, ci, res


def make_scatter(df, overall_stat):
    """散佈圖：四條件不同顏色／標記 + 整體迴歸線 + 95% 信賴區間。"""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for c in COND:
        sub = df[df["condition"] == c]
        ax.scatter(sub["conf_mean"], sub["gain"], s=26, alpha=0.75,
                   color=COND_COLOR[c], marker=COND_MARKER[c],
                   edgecolors="white", linewidths=0.4,
                   label=f"{COND_LABEL[c]} (n={len(sub)})", zorder=3)

    x_grid, y_hat, ci, res = ols_ci_band(df["conf_mean"].to_numpy(), df["gain"].to_numpy())
    ax.plot(x_grid, y_hat, color=C_INK, lw=1.8, zorder=4,
            label=f"OLS fit (r={overall_stat['pearson_r']:.3f}, "
                  f"p={overall_stat['pearson_p']:.4f})")
    ax.fill_between(x_grid, y_hat - ci, y_hat + ci, color=C_INK, alpha=0.12,
                     zorder=2, label="95% CI (mean response)")

    ax.axhline(0, color="#4a5a63", lw=0.8, ls=":")
    ax.set_xlabel("Per-image mean alignment confidence (conf_mean)")
    ax.set_ylabel("Reference gain (ref_miou − noref_miou), mIoU pp")
    ax.set_title("Alignment confidence vs. reference gain (n=%d)" % len(df),
                 fontsize=12, color=C_INK, pad=10)
    ax.legend(loc="best", frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "conf_gain_scatter.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    df_raw = pd.read_csv(CSV_PATH)
    n_before = len(df_raw)
    df = df_raw.dropna(subset=["conf_mean", "gain"]).copy()
    n_dropped = n_before - len(df)

    # 1) 整體與逐條件相關係數（同時即為第 4 項「條件內」相關係數）
    overall_stat = correlate(df, "overall")
    per_cond_stats = [correlate(g, c) for c, g in df.groupby("condition")]
    corr_rows = [overall_stat] + per_cond_stats

    # 2) 散佈圖
    scatter_path = make_scatter(df, overall_stat)

    # 3) 置信度三等分分箱
    df["conf_bin"] = pd.qcut(df["conf_mean"], 3, labels=["低", "中", "高"])
    binned = df.groupby("conf_bin", observed=True)["gain"].agg(
        ["count", "mean", "std"]).round(4)
    binned_out = {
        str(idx): dict(count=int(row["count"]), mean_gain=float(row["mean"]),
                       std_gain=float(row["std"]))
        for idx, row in binned.iterrows()
    }

    # 4) 條件混淆檢查：條件內相關是否仍顯著（判準邏輯，供人工覆核）
    within_cond_significant = [s["significant"] for s in per_cond_stats]
    within_cond_positive = [s["pearson_r"] > 0 for s in per_cond_stats]
    if overall_stat["significant"] and all(within_cond_significant) and \
            all(within_cond_positive) and overall_stat["pearson_r"] > 0:
        verdict = "a"
    elif overall_stat["significant"] and not (
            all(within_cond_significant) and all(within_cond_positive)):
        verdict = "b"
    else:
        verdict = "c"

    stats_out = dict(
        n_total=n_before,
        n_dropped_na=n_dropped,
        significance_alpha=SIG_ALPHA,
        correlations=corr_rows,
        confidence_bins=binned_out,
        verdict=verdict,
        verdict_note={
            "a": "整體與條件內皆顯著正相關 → 對齊品質決定參考效益，因果鏈成立。",
            "b": "僅整體顯著、條件內不顯著（或方向不一致）→ 條件與置信度共變，無法分離。",
            "c": "無顯著相關 → 參考效益的條件依賴另有來源，4.5.2 節敘述需修正。",
        }[verdict],
    )

    json_path = os.path.join(OUT_DIR, "conf_gain_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"排除缺值：{n_dropped} / {n_before} 筆（conf_mean 或 gain 為 NaN）")
    print("整體：", overall_stat)
    for s in per_cond_stats:
        print(f"  {s['scope']:>6s}：", s)
    print("分箱：", binned_out)
    print(f"判準結果：({verdict}) {stats_out['verdict_note']}")
    print("輸出：")
    print("  -", scatter_path)
    print("  -", json_path)


if __name__ == "__main__":
    main()
