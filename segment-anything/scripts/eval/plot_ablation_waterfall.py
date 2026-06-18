#!/usr/bin/env python
"""消融累積瀑布圖：從 baseline(R1) 逐模組疊加到 FULL(R7)，標出每步 Δ。

純 CPU，只讀各 run 的 e1_results.json（overall_miou）。
缺的累積中間列（Batch 2 的 R2–R5）會自動跳過並在圖上註記，補齊後重跑即完整。
FULL 採 seed-matched 的 R7_seed42（與 batch1 報告口徑一致）。
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation')
SAVE = os.path.join(OUT_ROOT, 'fig_ablation_waterfall.png')

# 累積路徑（每列相對前列只加一個模組），seed=42；FULL=R7_seed42
CUMULATIVE = [
    ('R1',   'R1_seed42',   'baseline\n(bare SAM+per-class+CE)'),
    ('R2',   'R2_seed42',   '+Ref (post)'),
    ('R3',   'R3_seed42',   'post->pre'),
    ('R4',   'R4_seed42',   '+unified decoder'),
    ('R5',   'R5_seed42',   '+LRH'),
    ('R6',   'R6_seed42',   '+Lovasz/Dice'),
    ('FULL', 'R7_seed42',   '+MFB (FULL)'),
]


def load_miou(run_dir):
    p = os.path.join(OUT_ROOT, run_dir, 'e1_results.json')
    if not os.path.isfile(p):
        return None
    return json.load(open(p))['overall_miou'] * 100.0


def main():
    present, missing = [], []
    for tag, d, label in CUMULATIVE:
        v = load_miou(d)
        (present if v is not None else missing).append((tag, label, v))

    # 防呆：累積瀑布圖必須 R1–R7 全到齊才誠實。缺中間列時若硬畫，
    # 會把缺的列的增益合併到下一個存在的步驟，造成錯誤歸因（捏造）。
    # 故缺任一列即跳過，不產生半套誤導圖；Batch 2 補齊後重跑自動啟用。
    if missing:
        raise SystemExit(
            '⏭️  累積瀑布圖跳過：缺 %s（Batch 2）。'
            '\n   累積圖需 R1–R7 全到齊才不會誤導歸因；'
            '請改看 fig_ablation_batch1_contrib.png（Batch 1 誠實版）。'
            % ', '.join(t for t, _, _ in missing)
        )

    tags   = [t for t, _, _ in present]
    labels = [l for _, l, _ in present]
    vals   = [v for _, _, v in present]

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(present)), 6))

    # 第一根（baseline）從 0 起；其後每根從前一個水準浮起，顯示增量
    base = 0.0
    for i, (tag, v) in enumerate(zip(tags, vals)):
        if i == 0:
            ax.bar(i, v, bottom=0, color='#4C72B0', width=0.6)
            ax.text(i, v + 0.6, f'{v:.1f}', ha='center', fontweight='bold')
        else:
            delta = v - vals[i - 1]
            color = '#55A868' if delta >= 0 else '#C44E52'
            ax.bar(i, abs(delta), bottom=min(vals[i - 1], v), color=color, width=0.6)
            # 連接線
            ax.plot([i - 1 + 0.3, i - 0.3], [vals[i - 1], vals[i - 1]],
                    color='gray', ls='--', lw=0.8)
            sign = '+' if delta >= 0 else ''
            ax.text(i, max(vals[i - 1], v) + 0.6, f'{sign}{delta:.1f}',
                    ha='center', fontweight='bold',
                    color='#2A6F3B' if delta >= 0 else '#8B2E33')
            ax.text(i, v - 2.0, f'{v:.1f}', ha='center', fontsize=9, color='white')

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f'{t}\n{l}' for t, l in zip(tags, labels)], fontsize=8)
    ax.set_ylabel('Overall mIoU (%)')
    ax.set_ylim(0, max(vals) + 8)
    title = 'Ablation Cumulative Waterfall (seed=42; FULL=R7_seed42=%.1f)' % vals[-1]
    if missing:
        title += '\n[missing %s - rerun after Batch 2]' % ', '.join(t for t, _, _ in missing)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE, dpi=150)
    print(f'✅ 瀑布圖 → {SAVE}')
    if missing:
        print('⚠️ 缺累積中間列：', ', '.join(t for t, _, _ in missing))


if __name__ == '__main__':
    main()
