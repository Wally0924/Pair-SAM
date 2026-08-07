#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""動態物體面積與跨視角對齊置信度的關係（論文第 4.4 節）。

核心結論：動態物體占畫面的比例越大，該區域的對齊置信度越低。
因此依像素數加權的動態區域平均（0.433）由大面積動態樣本主導，
低於靜態區域的 0.570；若改以逐張未加權平均，兩者順序會反轉。
本圖把加權來源畫成可見的幾何事實：橫軸即權重來源。

輸入：figures_defense/reference_gain/per_image.csv（406 列，ACDC val）
輸出：Images/ch4_conf_dynamic_area.png（300 dpi）與同名 .pdf

用法：
    conda run -n sam_env python scripts/eval/plot_conf_dynamic_area.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))              # segment-anything/
REPO = os.path.abspath(os.path.join(ROOT, ".."))
CSV_PATH = os.path.join(ROOT, "figures_defense", "reference_gain", "per_image.csv")
THESIS = os.path.join(REPO, "paper", "Chinese_master_thesis")
OUT_BASE = os.path.join(THESIS, "Images", "ch4_conf_dynamic_area")

# 論文正文引用的兩個像素加權統計量（由本腳本重算並斷言一致）
REF_STATIC = 0.570
REF_DYNAMIC = 0.433

COND = ["fog", "rain", "snow", "night"]
COND_LABEL = {"fog": "霧天", "rain": "雨天", "snow": "雪天", "night": "夜間"}
# 低飽和度配色：四種天候為中性族群，基準線以深灰承擔訊號角色
COND_COLOR = {"fog": "#5b8fa8", "rain": "#4f8a6b", "snow": "#c08a3e", "night": "#a85a52"}
COND_MARKER = {"fog": "o", "rain": "s", "snow": "^", "night": "D"}
C_INK = "#16222a"
C_LINE = "#7a3b34"


def load_font():
    """優先使用論文內文字型（標楷體），使圖內中文與正文一致。"""
    path = os.path.join(THESIS, "DFKai-SB.ttf")
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path).get_name()
    for name in ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    raise SystemExit("找不到可用的中文字型，無法產生中文軸標")


def read_rows():
    """回傳 (可繪製列, 全集列數, 排除列數, 全集像素加權統計)。

    兩條基準線以「全集 406 張」計算，與正文 4.4 節的統計範圍一致；
    散布點則只能畫有動態像素的影像（對數橫軸需要正值）。
    """
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    kept, dropped = [], 0
    ws = ns = wd = nd = 0.0
    for r in rows:
        n_sta = int(r["n_static"] or 0)
        if n_sta > 0 and r["conf_static"]:
            ws += float(r["conf_static"]) * n_sta
            ns += n_sta
        try:
            n_dyn = int(r["n_dynamic"])
            conf_dyn = float(r["conf_dynamic"])
        except (ValueError, KeyError):
            dropped += 1
            continue
        if n_dyn <= 0:
            dropped += 1
            continue
        wd += conf_dyn * n_dyn
        nd += n_dyn
        kept.append({
            "cond": r["condition"],
            "frac": 100.0 * n_dyn / int(r["n_valid"]),
            "conf_dyn": conf_dyn,
        })
    return kept, len(rows), dropped, ws / ns, wd / nd


def main():
    cjk = load_font()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cjk, "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        # 標楷體缺 U+2212，數學文字改由 DejaVu 承擔，避免負號變成方框
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 11,
        "axes.unicode_minus": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": C_INK,
        "legend.frameon": False,
    })

    rows, n_total, n_dropped, w_static, w_dyn = read_rows()
    x = np.array([r["frac"] for r in rows])
    y = np.array([r["conf_dyn"] for r in rows])

    assert abs(w_dyn - REF_DYNAMIC) < 0.005, f"動態加權值 {w_dyn:.4f} 與正文 {REF_DYNAMIC} 不符"
    assert abs(w_static - REF_STATIC) < 0.005, f"靜態加權值 {w_static:.4f} 與正文 {REF_STATIC} 不符"

    # 對數座標的正值保證：read_rows 已排除 n_dynamic <= 0，此處明示斷言
    assert x.min() > 0, "動態像素占比必須為正值才能取對數"
    lx = np.log10(x)
    r_p, p_p = stats.pearsonr(lx, y)
    slope, intercept = np.polyfit(lx, y, 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for c in COND:
        m = [i for i, r in enumerate(rows) if r["cond"] == c]
        ax.scatter(x[m], y[m], s=22, marker=COND_MARKER[c], c=COND_COLOR[c],
                   alpha=0.62, linewidths=0.4, edgecolors="white",
                   label=f"{COND_LABEL[c]}（{len(m)} 張）", zorder=3)

    # 迴歸線與 95% 信賴帶
    gx = np.linspace(lx.min(), lx.max(), 200)
    gy = slope * gx + intercept
    resid = y - (slope * lx + intercept)
    se = np.sqrt(np.sum(resid ** 2) / (len(y) - 2))
    ci = 1.96 * se * np.sqrt(1.0 / len(y)
                             + (gx - lx.mean()) ** 2 / np.sum((lx - lx.mean()) ** 2))
    ax.fill_between(10 ** gx, gy - ci, gy + ci, color=C_LINE, alpha=0.13, lw=0, zorder=2)
    ax.plot(10 ** gx, gy, color=C_LINE, lw=1.8, zorder=4)

    # 兩條像素加權基準線，直接標註取代圖例
    ax.axhline(w_static, color=C_INK, lw=1.1, ls="-", alpha=0.75, zorder=1)
    ax.axhline(w_dyn, color=C_INK, lw=1.1, ls="--", alpha=0.75, zorder=1)
    # 直接標註置於左下空白區，避開右側密集點雲
    ax.text(x.min() * 1.15, w_static + 0.02, f"靜態區域像素加權平均 {w_static:.3f}",
            ha="left", va="bottom", fontsize=9.5, color=C_INK)
    ax.text(x.min() * 1.15, w_dyn - 0.02, f"動態區域像素加權平均 {w_dyn:.3f}",
            ha="left", va="top", fontsize=9.5, color=C_INK)

    ax.set_xscale("log")
    # 以十進位標籤取代 10^{-2} 形式，兼顧可讀性與缺字問題
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2f}".rstrip("0")))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("動態物體像素占影像有效像素之比例（%）")
    ax.set_ylabel("動態區域平均對齊置信度")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="major", axis="both", ls=":", lw=0.6, color="#c8d0d4", zorder=0)
    ax.set_axisbelow(True)

    expo = int(np.floor(np.log10(p_p)))
    mant = p_p / 10 ** expo
    ax.text(0.025, 0.05,
            f"Pearson $r$ = {r_p:.3f}"
            f"（$p$ = ${mant:.1f}\\times10^{{{expo}}}$，$n$ = {len(y)}）",
            transform=ax.transAxes, fontsize=10, color=C_INK, va="bottom")
    ax.legend(loc="upper right", fontsize=9.5, handletextpad=0.4,
              borderaxespad=0.6, labelspacing=0.35)

    fig.tight_layout()
    # 論文以 PNG 置入（與既有 ch4 圖一致），另存向量檔供後續編修
    fig.savefig(OUT_BASE + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"樣本：全集 {n_total} 張，排除無動態像素 {n_dropped} 張，實際繪製 {len(y)} 張")
    print(f"像素加權：靜態 {w_static:.4f}｜動態 {w_dyn:.4f}")
    print(f"迴歸：斜率 {slope:.4f}／十倍面積，Pearson r = {r_p:.3f}，p = {p_p:.2e}")
    print(f"輸出：{OUT_BASE}.png / .pdf")


if __name__ == "__main__":
    main()
