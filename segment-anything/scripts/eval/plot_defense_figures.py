#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""口試簡報用消融圖表繪製腳本。

讀取 outputs_ablation_m2f/<RUN>/e1_results.json（pixel decoder 修復後的 M-series
乾淨結果），產生口試簡報所需的核心圖，輸出到 figures_defense/。

所有數值直接來自論文第四章採用的同一批 run，不重新推論、不重新評測。
標籤一律使用英文（Fog/Rain/Snow/Night），避免字型相依。

用法：
    python scripts/eval/plot_defense_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # segment-anything/
ABL = os.path.join(ROOT, "outputs_ablation_m2f")
OUT = os.path.join(ROOT, "figures_defense")
os.makedirs(OUT, exist_ok=True)

# 依論文採用的 run 對應（semB / confmod / noext 為重跑定案版本）
RUN = {
    "B0": "B0_seed42",            # 解碼端微調基準（無 Adapter）
    "B1": "B1_semB_seed42",       # + Adapter（無參考、無條件 token）
    "B2": "B2_seed42",            # + 參考影像
    "FULL": "FULL_seed42",        # + 條件 token（完整模型）
    "W1": "W1_seed42",            # 移除 Adapter
    "W2": "W2_semB_seed42",       # 移除參考影像
    "W3": "W3_confmod_seed42",    # 移除置信度調變
    "W4": "W4_seed42",            # 同影像注入基線
    "W5": "W5_seed42",            # 移除來源域預訓練
    "W6": "W6_noext_seed42",      # 移除抽取器（單向）
    "T1": "T1_seed42",            # 僅訓練 Adapter
    "T3": "T3_seed42",            # 全解凍
    "FULL_s1": "FULL_seed1234",
    "FULL_s2": "FULL_seed2026",
}

COND = ["fog", "rain", "snow", "night"]
COND_LABEL = ["Fog", "Rain", "Snow", "Night"]

# 一致的配色（與簡報霧藍/琥珀/綠/紅呼應）
C_MIST = "#3a7ea3"
C_AMBER = "#c07526"
C_GREEN = "#4a8a68"
C_RED = "#b0473d"
C_GREY = "#9aa7ad"
C_INK = "#16222a"

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


def load(run_key):
    p = os.path.join(ABL, RUN[run_key], "e1_results.json")
    with open(p) as f:
        return json.load(f)


def miou(run_key):
    return load(run_key)["overall_miou"] * 100.0


def cond_miou(run_key):
    d = load(run_key)["per_condition_miou"]
    return [d[c] * 100.0 for c in COND]


def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.4, color=C_INK, size=9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=size, color=color)


# ---------------------------------------------------------------------------
# 圖 1：累積式消融（B0 → B1 → B2 → FULL）
# ---------------------------------------------------------------------------
def fig_cumulative():
    keys = ["B0", "B1", "B2", "FULL"]
    labels = ["Decoder FT\n(no Adapter)", "+ Adapter", "+ Reference", "+ Cond token\n(Full)"]
    vals = [miou(k) for k in keys]
    deltas = [None] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    colors = [C_GREY, C_MIST, C_MIST, C_GREEN]
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    _bar_labels(ax, bars)
    for i, d in enumerate(deltas):
        if d is None:
            continue
        sign = "+" if d >= 0 else ""
        ax.annotate(f"{sign}{d:.2f}", xy=(i, vals[i] + 1.4), ha="center",
                    fontsize=9, color=(C_GREEN if d >= 0 else C_RED), fontweight="bold")
    ax.set_ylim(70, 78)
    ax.set_ylabel("ACDC val mIoU (%)")
    ax.set_title("Cumulative ablation: what actually helps", fontsize=12, color=C_INK, pad=10)
    fig.tight_layout()
    out = os.path.join(OUT, "01_cumulative_ablation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 圖 2：逐項移除 Δ（相對完整模型），水平長條、正負分色
# ---------------------------------------------------------------------------
def fig_leave_one_out():
    full = miou("FULL")
    items = [
        ("Remove source pretrain (W5)", miou("W5") - full),
        ("Remove Adapter (W1)", miou("W1") - full),
        ("Remove conf. modulation (W3)", miou("W3") - full),
        ("Remove extractor / 1-way (W6)", miou("W6") - full),
        ("Remove reference (W2)", miou("W2") - full),
        ("Same-image baseline (W4)", miou("W4") - full),
    ]
    items.sort(key=lambda x: x[1])  # 最負在下
    labels = [x[0] for x in items]
    deltas = [x[1] for x in items]
    colors = [C_RED if d < 0 else C_GREEN for d in deltas]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.barh(labels, deltas, color=colors, zorder=3, height=0.62)
    ax.axvline(0, color="#4a5a63", lw=1.0)
    for b, d in zip(bars, deltas):
        x = b.get_width()
        ax.text(x + (0.12 if x >= 0 else -0.12), b.get_y() + b.get_height() / 2,
                f"{'+' if d >= 0 else ''}{d:.2f}", va="center",
                ha="left" if x >= 0 else "right", fontsize=9,
                color=(C_GREEN if d >= 0 else C_RED), fontweight="bold")
    ax.set_xlabel("Δ mIoU vs. Full model (%)  —  seed std ≈ 0.14")
    ax.set_title("Leave-one-out ablation (relative to Full = %.2f%%)" % full,
                 fontsize=12, color=C_INK, pad=10)
    ax.set_xlim(min(deltas) - 1.8, max(deltas) + 1.8)
    fig.tight_layout()
    out = os.path.join(OUT, "02_leave_one_out_ablation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 圖 3：跨視角參考的逐條件淨貢獻（FULL - W2，正雨雪、負霧夜）
# ---------------------------------------------------------------------------
def fig_reference_gain_by_condition():
    full_c = cond_miou("FULL")
    w2_c = cond_miou("W2")           # 移除參考
    gain = [full_c[i] - w2_c[i] for i in range(4)]
    overall = miou("FULL") - miou("W2")

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    colors = [C_GREEN if g >= 0 else C_RED for g in gain]
    bars = ax.bar(COND_LABEL, gain, color=colors, width=0.6, zorder=3)
    ax.axhline(0, color="#4a5a63", lw=1.0)
    ax.axhline(overall, color=C_AMBER, lw=1.4, ls="--",
               label=f"Overall avg {overall:+.2f}")
    for b, g in zip(bars, gain):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (0.12 if h >= 0 else -0.28),
                f"{g:+.2f}", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=10, color=(C_GREEN if g >= 0 else C_RED), fontweight="bold")
    ax.set_ylabel("Reference net gain (Full − no-ref)  Δ mIoU (%)")
    ax.set_title("Cross-view reference: gain depends on condition",
                 fontsize=12, color=C_INK, pad=10)
    ax.set_ylim(min(gain) - 1.2, max(gain) + 1.2)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT, "03_reference_gain_by_condition.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 圖 4：訓練策略取捨（僅 Adapter / 解碼端微調 / 全解凍）vs 可訓練參數比例
# ---------------------------------------------------------------------------
def fig_training_strategy():
    keys = ["T1", "FULL", "T3"]
    labels = ["Adapter only\n(~2.1%)", "Decoder FT\n(~4.7%, Full)", "Unfreeze all\n(~80%)"]
    vals = [miou(k) for k in keys]
    ratios = [2.1, 4.7, 80.0]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    colors = [C_MIST, C_GREEN, C_GREY]
    bars = ax.bar(labels, vals, color=colors, width=0.6, zorder=3)
    _bar_labels(ax, bars, dy=0.25)
    upper = vals[-1]
    for i, v in enumerate(vals):
        pct = v / upper * 100
        ax.text(i, v - 1.6, f"{pct:.1f}% of\nupper bound", ha="center",
                va="top", fontsize=8, color="#4a5a63")
    ax.set_ylim(72, 82)
    ax.set_ylabel("ACDC val mIoU (%)")
    ax.set_title("Freezing trade-off: 4.7% params keep 94.7% of full-unfreeze",
                 fontsize=11.5, color=C_INK, pad=10)
    fig.tight_layout()
    out = os.path.join(OUT, "04_training_strategy_tradeoff.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 圖 5：三組隨機種子穩健性（逐條件 + overall，附誤差條）
# ---------------------------------------------------------------------------
def fig_seed_robustness():
    seeds = ["FULL", "FULL_s1", "FULL_s2"]
    all_c = np.array([cond_miou(s) for s in seeds])       # (3,4)
    all_o = np.array([miou(s) for s in seeds])            # (3,)
    means = list(all_c.mean(axis=0)) + [all_o.mean()]
    stds = list(all_c.std(axis=0, ddof=1)) + [all_o.std(ddof=1)]
    labels = COND_LABEL + ["Overall"]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    colors = [C_MIST, C_MIST, C_MIST, C_MIST, C_GREEN]
    bars = ax.bar(labels, means, yerr=stds, color=colors, width=0.6, zorder=3,
                  capsize=5, error_kw=dict(ecolor="#4a5a63", lw=1.2))
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.5,
                f"{m:.2f}\n±{s:.2f}", ha="center", va="bottom", fontsize=8.5,
                color=C_INK)
    ax.set_ylim(45, 90)
    ax.set_ylabel("ACDC val mIoU (%)")
    ax.set_title("Seed robustness: 3 seeds, overall std = %.2f" % stds[-1],
                 fontsize=12, color=C_INK, pad=10)
    fig.tight_layout()
    out = os.path.join(OUT, "05_seed_robustness.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 圖 6：逐條件難度（完整模型 4 條件 mIoU，強調夜間最難）
# ---------------------------------------------------------------------------
def fig_condition_difficulty():
    vals = cond_miou("FULL")
    order = sorted(range(4), key=lambda i: -vals[i])
    labels = [COND_LABEL[i] for i in order]
    v = [vals[i] for i in order]
    colors = [C_GREEN if x == max(v) else (C_RED if x == min(v) else C_MIST) for x in v]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, v, color=colors, width=0.58, zorder=3)
    _bar_labels(ax, bars, dy=0.4)
    ax.set_ylim(45, 85)
    ax.set_ylabel("ACDC val mIoU (%)")
    ax.set_title("Per-condition difficulty (Full model)", fontsize=12, color=C_INK, pad=10)
    fig.tight_layout()
    out = os.path.join(OUT, "06_per_condition_difficulty.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 圖 7：零初始化閘控 γ 隨 epoch 成長 + 注入 cosine 相似度
#        （用碩論定稿 FULL_seed42 之 train_log.csv，30 epoch、γ 自 ≈0 起步，
#         與「零初始化」敘述一致；取代會議版 ch4_gates_cosine.png）
# ---------------------------------------------------------------------------
def fig_gate_growth():
    import csv
    p = os.path.join(ABL, RUN["FULL"], "train_log.csv")
    rows = list(csv.DictReader(open(p)))

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, ValueError):
                out.append(np.nan)
        return np.array(out)

    ep = col("epoch")
    gates = {
        "s0": col("train_inject_gate_s0"),
        "s1": col("train_inject_gate_s1"),
        "s2": col("train_inject_gate_s2"),
        "s3": col("train_inject_gate_s3"),
    }
    cos = col("train_inject_cos_sim")
    gcolors = {"s0": C_MIST, "s1": C_GREEN, "s2": C_AMBER, "s3": C_RED}

    fig, (axg, axc) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) 閘控成長
    for k, v in gates.items():
        axg.plot(ep, v, color=gcolors[k], lw=1.8, label="gate %s" % k)
    axg.axhline(0.0, color="#4a5a63", lw=0.8, ls=":")
    axg.set_xlabel("Epoch")
    axg.set_ylabel("Injection gate value (γ)")
    axg.set_title("(a) Zero-init gate grows from ≈0", fontsize=12, color=C_INK, pad=8)
    axg.legend(frameon=False, fontsize=9, ncol=2)
    axg.set_xlim(1, ep[-1])

    # (b) 注入 cosine 相似度
    axc.plot(ep, cos, color=C_MIST, lw=2.0)
    axc.set_xlabel("Epoch")
    axc.set_ylabel("cos(q, q + γ·Δ)")
    axc.set_title("(b) Injection cosine similarity", fontsize=12, color=C_INK, pad=8)
    axc.set_xlim(1, ep[-1])

    fig.suptitle("Zero-initialized gate telemetry (thesis FULL model, %d epochs)"
                 % int(ep[-1]), fontsize=11, color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, "07_gate_growth.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    made = []
    made.append(fig_cumulative())
    made.append(fig_leave_one_out())
    made.append(fig_reference_gain_by_condition())
    made.append(fig_training_strategy())
    made.append(fig_seed_robustness())
    made.append(fig_condition_difficulty())
    made.append(fig_gate_growth())
    print("=" * 60)
    print(f"完成，共 {len(made)} 張圖，輸出於 {OUT}")
    for m in made:
        print("  -", os.path.basename(m))


if __name__ == "__main__":
    main()
