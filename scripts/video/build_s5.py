"""S5 — 參數效率收尾(96–112 s)。

兩拍:(a) 三張數字卡 (b) ACDC test 四條件對照。
數字全部取自 root.tex 的 tab:per_condition 與 tab:strategy。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from style import (W, H, FPS, canvas, draw_title, draw_subtitle, draw_note,
                   save_seq, ease, font, fig_to_image, dark_axes, COLORS)

ROOT = os.path.expanduser('~/SAM_research-1')
OUT = f'{ROOT}/docs/video/segments/s5'
BACKDROP = f'{ROOT}/segment-anything/figures_defense/04_training_strategy_tradeoff.png'

CARDS = [
    ('72.1%', 'mIoU on the ACDC test set', COLORS['Pair-SAM']),
    ('4.75%', 'of the parameters trained  ·  40.2 M', COLORS['CMA']),
    ('ViT-H', 'frozen throughout', COLORS['Refign']),
]

CONDS = ['fog', 'rain', 'snow', 'night']
SERIES = [
    ('DeepLabv3+  (fully fine-tuned)', [69.1, 74.1, 69.6, 60.9], '#8a8a8a'),
    ('HRNet  (fully fine-tuned)', [74.7, 77.7, 76.3, 65.3], COLORS['Refign']),
    ('Pair-SAM  (4.75% trained)', [71.2, 77.6, 74.2, 61.9], COLORS['Pair-SAM']),
]

SUB_A = '72.1% mIoU on ACDC test, training only 4.75% of the parameters.'
SUB_B = 'Ahead of a fully fine-tuned DeepLabv3+ in every condition.'

BEAT_A, BEAT_B = 210, 270          # 7s + 9s = 16s
FADE = 12


def backdrop():
    """純深色背景。

    原計畫以 04_training_strategy_tradeoff.png 作低透明度襯底,實測後移除:
    該圖自帶 74.56 / 76.02 / 80.26 三個數值,與前景的 72.1 / 4.75 併置時
    會讓觀眾讀到互相矛盾的數字。
    """
    return canvas()


_BD = None


def cards_frame(n_shown, t_in):
    global _BD
    if _BD is None:
        _BD = backdrop()
    c = _BD.copy()
    d = ImageDraw.Draw(c, 'RGBA')
    y = 168
    for i, (big, small, col) in enumerate(CARDS):
        if i > n_shown:
            continue
        a = 1.0 if i < n_shown else ease(t_in)
        fb, fs = font(58, True), font(19)
        wb = d.textlength(big, font=fb)
        ws = d.textlength(small, font=fs)
        total = wb + 24 + ws
        x = (W - total) / 2
        v = int(255 * a)
        d.text((x, y + i * 96), big, fill=tuple(int(col[k:k + 2], 16) * v // 255
                                                for k in (1, 3, 5)), font=fb)
        d.text((x + wb + 24, y + i * 96 + 30), small, fill=(v, v, v), font=fs)
    draw_title(d, 'Adaptation on a fixed budget')
    draw_subtitle(d, SUB_A)
    draw_note(d, 'The fully unfrozen upper bound is 80.26 on ACDC val; '
                 'the full model retains 94.7% of it with 4.75% of the parameters.')
    return c


def conds_plot():
    fig, ax = plt.subplots(figsize=(10.6, 4.0), dpi=100)
    dark_axes(ax, fig)
    xs = np.arange(len(CONDS))
    w = 0.26
    for i, (label, vals, col) in enumerate(SERIES):
        ax.bar(xs + (i - 1) * w, vals, width=w, color=col, label=label)
        for x, v in zip(xs + (i - 1) * w, vals):
            ax.text(x, v + 0.6, f'{v:.1f}', ha='center', va='bottom',
                    color=col, fontsize=10, weight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels([c.upper() for c in CONDS], color='#dddddd', fontsize=13)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(55, 84)
    ax.set_ylabel('mIoU (%)  ·  ACDC test', color='#aaaaaa', fontsize=11)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    leg = ax.legend(loc='upper right', fontsize=9, facecolor='#1e1e1e',
                    edgecolor='#3a3a3a', framealpha=0.95, ncol=1)
    for t in leg.get_texts():
        t.set_color('#dddddd')
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.14)
    img = fig_to_image(fig)
    plt.close(fig)
    return img


def conds_frame():
    c = canvas()
    d = ImageDraw.Draw(c, 'RGBA')
    c.paste(conds_plot(), (100, 116))
    draw_title(d, 'ACDC test  ·  per condition  ·  2,000 images')
    draw_subtitle(d, SUB_B)
    draw_note(d, 'In rain, 77.6 is within 0.1 points of HRNet, which fine-tunes its entire network.')
    return c


def main():
    frames = []
    step = BEAT_A // 3
    for i in range(3):
        for j in range(step):
            frames.append(cards_frame(i, min(1.0, j / 18)))
    frames += [cards_frame(2, 1.0)] * (BEAT_A - len(frames))

    cf = conds_frame()
    for i in range(FADE):
        frames.append(Image.blend(frames[BEAT_A - 1], cf, ease((i + 1) / FADE)))
    frames += [cf] * (BEAT_B - FADE)

    save_seq(frames, OUT)
    print(f'S5: {len(frames)} frames ({len(frames)/FPS:.2f}s) -> {OUT}')


if __name__ == '__main__':
    main()
