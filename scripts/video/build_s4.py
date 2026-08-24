"""S4 — 跨視角參考的真實價碼(82–96 s)。

畫相對「不注入參考」基線的差值,而非絕對 mIoU:絕對值需要 75–77 的截斷軸,
會把 0.87 分的落差放大成「條長少一半」,誇大自家的負面結果。
數值取自 root.tex tab:ablation。論文結論:調變把代價由 0.87 降到 0.48,
但未使參考在此 benchmark 上轉為淨增益 —— 本段照此呈現,不得反寫。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from style import (W, FPS, canvas, draw_title, draw_subtitle, draw_note,
                   save_seq, ease, fig_to_image, dark_axes, COLORS)

OUT = os.path.expanduser('~/SAM_research-1/docs/video/segments/s4')

BASELINE = 76.50                      # Without reference image
BARS = [                              # (標籤, 絕對 mIoU, Δ vs 基線, 顏色)
    ('Reference, unmodulated', 75.63, -0.87, COLORS['CMA']),
    ('Reference, confidence-modulated', 76.02, -0.48, COLORS['Pair-SAM']),
]
SEED_STD = 0.14

SUB = ('Injecting the reference at full strength costs 0.87 points. '
       'Modulation recovers part of it, not all.')

GROW, GAP1, GAP2, HOLD = 90, 90, 90, 150       # 3s + 3s + 3s + 5s = 14s


def plot(grow, show1, show2):
    fig, ax = plt.subplots(figsize=(10.2, 3.9), dpi=100)
    dark_axes(ax, fig)
    ax.grid(color='#2a2a2a', lw=0.6, axis='x')

    ys = np.arange(len(BARS))[::-1]
    shows = [show1, show2]
    for y, (label, absolute, delta, col), show in zip(ys, BARS, shows):
        v = delta * ease(grow)
        ax.barh(y, v, height=0.38, color=col)
        ax.text(0.035, y, label, ha='left', va='center', color='#f2f2f2', fontsize=14)
        ax.text(0.035, y - 0.24, f'{absolute:.2f} mIoU', ha='left', va='center',
                color='#8a8a8a', fontsize=11)
        if show > 0:
            a = ease(show)
            ax.text(v - 0.03, y, f'{delta:+.2f}', ha='right', va='center',
                    color=col, fontsize=17, weight='bold', alpha=a)

    # 基線與 seed 雜訊尺度
    ax.axvline(0, color='#dddddd', lw=1.4)
    for sgn in (-1, 1):
        ax.axvline(sgn * SEED_STD, color='#666666', lw=0.9, ls='--')
    ax.text(0, len(BARS) - 0.52, f'reference-free baseline  {BASELINE:.2f} mIoU',
            ha='center', va='bottom', color='#dddddd', fontsize=12)
    ax.text(-SEED_STD, -0.62, f'±{SEED_STD} seed noise', ha='center', va='bottom',
            color='#777777', fontsize=10)

    ax.set_xlim(-1.15, 0.62)
    ax.set_ylim(-0.72, len(BARS) - 0.22)
    ax.set_yticks([])
    ax.set_xlabel('change in mIoU against not injecting a reference at all (ACDC val)',
                  color='#aaaaaa', fontsize=11)
    for s in ('left', 'top', 'right'):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.18)
    img = fig_to_image(fig)
    plt.close(fig)
    return img


def frame(grow, show1, show2):
    c = canvas()
    d = ImageDraw.Draw(c, 'RGBA')
    c.paste(plot(grow, show1, show2), (130, 116))
    draw_title(d, 'What the cross-view reference actually costs  ·  ACDC validation split')
    draw_subtitle(d, SUB)
    draw_note(d, 'Both gaps exceed the 0.14 seed deviation, so neither is run-to-run noise.')
    return c


def main():
    frames = []
    for i in range(GROW):
        frames.append(frame(i / GROW, 0, 0))
    for i in range(GAP1):
        frames.append(frame(1.0, min(1.0, i / 20), 0))
    for i in range(GAP2):
        frames.append(frame(1.0, 1.0, min(1.0, i / 20)))
    frames += [frames[-1]] * HOLD
    save_seq(frames, OUT)
    print(f'S4: {len(frames)} frames ({len(frames)/FPS:.2f}s) -> {OUT}')


if __name__ == '__main__':
    main()
