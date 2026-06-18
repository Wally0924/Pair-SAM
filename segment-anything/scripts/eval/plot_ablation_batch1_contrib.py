#!/usr/bin/env python
"""Batch 1 誠實版總覽圖（無捏造、不需 Batch 2）。

左圖：5 個 Batch 1 run 的 overall mIoU（R1 baseline + 三個 leave-one-out 變體 + FULL）。
右圖：各組件貢獻 = FULL - 對應 LOO 變體（Reference / MFB / Lovász+Dice）。
全 seed=42，FULL=R7_seed42=67.3，與 Batch 1 報告口徑一致。
純 CPU，只讀 e1_results.json。所有文字為英文，無中文。
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation')
SAVE = os.path.join(OUT_ROOT, 'fig_ablation_batch1_contrib.png')

# overall 圖：(短標籤, run 目錄)，由低到高排列
OVERALL = [
    ('R1\n(baseline)',     'R1_seed42'),
    ('A2\n(w/o Ref)',      'A2_seed42'),
    ('C2\n(w/o MFB)',      'R6_seed42'),
    ('C1\n(w/o Lov/Dice)', 'C1_seed42'),
    ('FULL',               'R7_seed42'),
]
# 貢獻圖：(組件名, LOO 變體 run 目錄)，貢獻 = FULL - 變體
CONTRIB = [
    ('Reference',    'A2_seed42'),
    ('MFB',          'R6_seed42'),
    ('Lovasz+Dice',  'C1_seed42'),
]
FULL_DIR = 'R7_seed42'


def miou(run_dir):
    p = os.path.join(OUT_ROOT, run_dir, 'e1_results.json')
    if not os.path.isfile(p):
        raise SystemExit(f'找不到 {p}')
    return json.load(open(p))['overall_miou'] * 100.0


def main():
    full = miou(FULL_DIR)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={'width_ratios': [3, 2]})

    # ── 左：overall mIoU bars ──
    labels = [l for l, _ in OVERALL]
    vals = [miou(d) for _, d in OVERALL]
    colors = ['#B0B0B0', '#DD8452', '#DD8452', '#DD8452', '#4C72B0']
    bars = axL.bar(range(len(vals)), vals, color=colors, width=0.62)
    for i, v in enumerate(vals):
        axL.text(i, v + 0.8, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)
    axL.set_xticks(range(len(labels)))
    axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel('Overall mIoU (%)')
    axL.set_ylim(0, max(vals) + 8)
    axL.set_title('Batch 1 overall mIoU (seed=42)', fontsize=11)
    axL.grid(axis='y', alpha=0.3)

    # ── 右：component contribution (FULL - LOO variant) ──
    cnames = [c for c, _ in CONTRIB]
    deltas = [full - miou(d) for _, d in CONTRIB]
    cbars = axR.bar(range(len(deltas)), deltas, color='#55A868', width=0.55)
    for i, dv in enumerate(deltas):
        axR.text(i, dv + 0.12, f'+{dv:.1f}', ha='center', fontweight='bold', fontsize=10)
    axR.set_xticks(range(len(cnames)))
    axR.set_xticklabels(cnames, fontsize=10)
    axR.set_ylabel('mIoU gain (FULL - leave-one-out)')
    axR.set_ylim(0, max(deltas) + 2)
    axR.set_title('Component contribution', fontsize=11)
    axR.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Batch 1 ablation summary  (FULL = R7_seed42 = {full:.1f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(SAVE, dpi=150)
    print(f'✅ Batch 1 總覽圖 → {SAVE}')


if __name__ == '__main__':
    main()
