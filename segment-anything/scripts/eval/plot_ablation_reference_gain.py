#!/usr/bin/env python
"""分條件 reference 增益圖：A2(no-ref) vs FULL 每個天氣條件的 mIoU 落差。

核心訊息：reference 的效益高度依賴條件——fog 退化最重、增益最大；night 幾乎無用。
這是「無參考會變差」最誠實有力的量化呈現（取代難以從全幀看出的定性圖）。
純 CPU，只讀 e1_results.json 的 per_condition_miou。全 seed=42，FULL=R7_seed42。
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation')
SAVE = os.path.join(OUT_ROOT, 'fig_ablation_reference_gain.png')
CONDS = ['fog', 'rain', 'snow', 'night']


def per_cond(run_dir):
    p = os.path.join(OUT_ROOT, run_dir, 'e1_results.json')
    return json.load(open(p))['per_condition_miou']


def main():
    a2 = per_cond('A2_seed42')
    full = per_cond('R7_seed42')
    a2_v = [a2[c] * 100 for c in CONDS]
    full_v = [full[c] * 100 for c in CONDS]
    gains = [f - a for f, a in zip(full_v, a2_v)]

    x = np.arange(len(CONDS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, a2_v, w, label='A2 (no reference)', color='#DD8452')
    ax.bar(x + w / 2, full_v, w, label='FULL (with reference)', color='#4C72B0')

    for i in range(len(CONDS)):
        ax.text(x[i] - w / 2, a2_v[i] + 0.7, f'{a2_v[i]:.1f}', ha='center', fontsize=9)
        ax.text(x[i] + w / 2, full_v[i] + 0.7, f'{full_v[i]:.1f}', ha='center', fontsize=9)
        # 增益標註（置於兩柱上方中間）
        top = max(a2_v[i], full_v[i])
        ax.annotate(f'+{gains[i]:.1f}', xy=(x[i], top + 3.0), ha='center',
                    fontsize=11, fontweight='bold',
                    color='#2A6F3B' if gains[i] > 3 else '#777777')

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONDS], fontsize=11)
    ax.set_ylabel('mIoU (%)')
    ax.set_ylim(0, max(full_v) + 12)
    ax.set_title('Reference contribution is condition-dependent\n'
                 '(largest under heavy degradation: Fog; negligible at Night)',
                 fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE, dpi=150)
    print(f'✅ 分條件 reference 增益圖 → {SAVE}')


if __name__ == '__main__':
    main()
