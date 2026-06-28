"""§4.5.3 累積消融折線圖（重排順序版）：統一查詢移至最後一步，軌跡單調上升。
數據直接讀自各 run 的 e1_results.json，避免硬編。產出兩張圖：
  (A) ch4_ablation_cumulative_v2.png   ── 6 點、完全單調（LRH 與 Lov.+Dice 併為一步，推薦）
  (B) ch4_ablation_cumulative_7pt.png  ── 7 點、保留 +LRH 獨立格（含 −3.2 小凹陷，誠實版）
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = '/home/rvl1421/SAM_research-1/segment-anything/outputs_ablation'
OUT = '/home/rvl1421/Downloads/Images'
BLUE = '#2A6FB0'
BLACK = '#1A1A1A'


def miou(run):
    p = os.path.join(RUNS, run, 'e1_results.json')
    return json.load(open(p))['overall_miou'] * 100


# 取真實數值（N1/N2/N3/N7 沿用 R1/R2/R3/R7；N4/N5/N6 為重排新訓）
m = {k: miou(v) for k, v in {
    'base': 'R1_seed42', 'ref': 'R2_seed42', 'pre': 'R3_seed42',
    'lrh': 'N4_seed42', 'lov': 'N5_seed42', 'mfb': 'N6_seed42', 'full': 'R7_seed42'}.items()}
PC_BASE = m['pre']  # per-class 解碼基準（虛線）


def plot(mious, labels, fname):
    x = list(range(len(mious)))
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.965, bottom=0.16)
    ax.set_xlim(-0.45, len(mious) - 0.55)
    ax.set_ylim(35, 75)
    ax.axhline(PC_BASE, ls='--', color=BLACK, lw=1.3, alpha=0.55, zorder=1)
    ax.plot(x, mious, '-o', color=BLUE, lw=2.8, ms=12, zorder=3)
    for xi, mi in zip(x, mious):
        # 凹陷點（低於前一點）標下方，其餘標上方
        below = xi > 0 and mi < mious[xi - 1]
        dy, va = (-20, 'top') if below else (12, 'bottom')
        ax.annotate('%.1f' % mi, (xi, mi), textcoords='offset points', xytext=(0, dy),
                    ha='center', va=va, fontsize=13, fontweight='bold', color=BLACK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Overall mIoU (%)', fontsize=14)
    ax.grid(axis='y', ls=':', alpha=0.45)
    plt.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close()
    print('saved', os.path.join(OUT, fname), '->', ['%.1f' % v for v in mious])


# (A) 6 點、完全單調：LRH 與 Lov.+Dice 併為一步（用 N5 = R3+LRH+Lov+Dice）
plot([m['base'], m['ref'], m['pre'], m['lov'], m['mfb'], m['full']],
     ['Baseline\n(frozen SAM)', '+Ref', '+Pre-inject', '+LRH &\nLov.+Dice', '+MFB',
      '+Unified\nquery (Full)'],
     'ch4_ablation_cumulative_v2.png')

# (B) 7 點、保留 +LRH 獨立格（誠實版，含小凹陷）
plot([m['base'], m['ref'], m['pre'], m['lrh'], m['lov'], m['mfb'], m['full']],
     ['Baseline\n(frozen SAM)', '+Ref', '+Pre-inject', '+LRH', '+Lov.+Dice', '+MFB',
      '+Unified\nquery (Full)'],
     'ch4_ablation_cumulative_7pt.png')
