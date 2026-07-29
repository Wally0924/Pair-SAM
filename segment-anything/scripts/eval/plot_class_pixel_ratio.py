#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""統計 ACDC 訓練集 19 類的畫素佔比並繪圖（口試簡報用）。

讀 Datasets/acdc_adverse_ref_rgb_train.csv 的 gt_path（labelTrainIds，0..18，
255=ignore），累計各類別畫素數，算佔比，畫成 log 縮放長條圖。
純讀標註、CPU、不動模型。輸出至 segment-anything/figures_defense/。
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # segment-anything/
REPO = os.path.abspath(os.path.join(ROOT, ".."))                # repo root
CSV = os.path.join(REPO, "Datasets", "acdc_adverse_ref_rgb_train.csv")
OUT = os.path.join(ROOT, "figures_defense")
os.makedirs(OUT, exist_ok=True)

# trainId 0..18 → 類別名（Cityscapes 標準）
CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]
IGNORE = 255

C_MIST = "#3a7ea3"
C_AMBER = "#c07526"
C_RED = "#b0473d"
C_INK = "#16222a"

# 使用系統 Noto Sans CJK（含中文），直接註冊字型檔避免名稱匹配失敗
import matplotlib.font_manager as fm
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_CJK).get_name()
plt.rcParams.update({
    "axes.unicode_minus": False,
    "axes.edgecolor": "#b8c6cb",
    "axes.grid": True, "grid.color": "#e3eaed", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 160,
})


def main():
    counts = np.zeros(19, dtype=np.int64)
    rows = list(csv.DictReader(open(CSV)))
    n = len(rows)
    print(f"統計 {n} 張 ACDC 訓練標註 …")
    for i, r in enumerate(rows):
        p = r["gt_path"]
        if not os.path.exists(p):
            continue
        arr = np.asarray(Image.open(p))
        valid = arr[arr != IGNORE]
        bc = np.bincount(valid.ravel(), minlength=19)[:19]
        counts += bc
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{n}")

    total = counts.sum()
    ratio = counts / total * 100.0  # 百分比

    # 依佔比排序（大→小）
    order = np.argsort(-ratio)
    names = [CLASSES[k] for k in order]
    vals = ratio[order]

    # 顏色：頭部藍、尾部（<0.1%）琥珀、最小紅
    colors = []
    for v in vals:
        if v < 0.05:
            colors.append(C_RED)
        elif v < 0.5:
            colors.append(C_AMBER)
        else:
            colors.append(C_MIST)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bars = ax.bar(names, vals, color=colors, width=0.72, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Pixel share  ·  log scale")
    ax.set_title("ACDC training set: long-tailed class pixel distribution "
                 "(19 classes)", fontsize=12.5, color=C_INK, pad=10)
    ax.set_ylim(vals.min() * 0.5, vals.max() * 1.6)
    # 不標具體百分比（數字以論文正文為準，避免圖文不一致）；僅標頭尾定性
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9.5)

    # 頭尾定性標註（不寫具體數字，數字以論文正文為準）
    ax.annotate("頭部大面積結構",
                xy=(1, vals[1]), xytext=(2.2, vals[0] * 0.62),
                fontsize=10, color=C_MIST, ha="left")
    ax.annotate("長尾：rider 最少\n（罕見動態類別）",
                xy=(18, vals[-1]), xytext=(11.5, vals[-1] * 2.6),
                fontsize=10, color=C_RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    ax.text(0.99, 0.94, "頭尾差距達數千倍（對數軸）",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=C_INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7ead9", ec=C_AMBER, lw=1))

    fig.tight_layout()
    out = os.path.join(OUT, "08_class_pixel_ratio.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # 印出數值表供投影片/核對
    print("\n=== ACDC 訓練集 19 類畫素佔比（大→小）===")
    for k in order:
        print(f"  {CLASSES[k]:<15} {ratio[k]:8.4f}%   ({counts[k]:,} px)")
    print(f"\n完成 → {out}")


if __name__ == "__main__":
    main()
