#!/usr/bin/env python
"""全 19 類別 per-class IoU 分組柱狀圖：凸顯 MFB 對長尾類的效果。

純 CPU，只讀各 run 的 e1_results.json（per_class_iou_overall）。
預設比較 R1(baseline) / R6(=C2 no-MFB) / FULL(R7_seed42)，全 seed=42。
類別依像素頻率由高到低排列，長尾類（rider/moto/bike）落在右側，差距一目了然。
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation')
SAVE = os.path.join(OUT_ROOT, 'fig_ablation_allclass.png')

# (圖例標籤, run 目錄, 顏色)
RUNS = [
    ('R1 baseline',     'R1_seed42', '#B0B0B0'),
    ('C2 (no MFB)',     'R6_seed42', '#DD8452'),
    ('FULL (seed=42)',  'R7_seed42', '#4C72B0'),
]
# 由頭部→長尾大致排序（與正文敘述一致）
CLASS_ORDER = [
    'road', 'sky', 'building', 'vegetation', 'car', 'sidewalk', 'terrain',
    'pole', 'wall', 'fence', 'traffic sign', 'traffic light', 'person',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'rider',
]


def load_pc(run_dir):
    p = os.path.join(OUT_ROOT, run_dir, 'e1_results.json')
    if not os.path.isfile(p):
        raise SystemExit(f'找不到 {p}')
    return json.load(open(p))['per_class_iou_overall']


def main():
    data = [(lab, load_pc(d), c) for lab, d, c in RUNS]
    classes = [c for c in CLASS_ORDER if c in data[0][1]]
    x = np.arange(len(classes))
    n = len(data)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (lab, pc, color) in enumerate(data):
        vals = [pc.get(c, 0.0) * 100 for c in classes]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=lab, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('IoU (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Per-class IoU, head to tail (long-tail classes on the right)',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    # 標記長尾區
    ax.axvspan(len(classes) - 3.5, len(classes) - 0.5, color='red', alpha=0.06)
    plt.tight_layout()
    plt.savefig(SAVE, dpi=150)
    print(f'✅ 全類別柱狀圖 → {SAVE}')


if __name__ == '__main__':
    main()
