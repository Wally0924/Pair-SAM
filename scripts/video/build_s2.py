"""S2 — 方法:凍結主幹 + 注入結構 + 兩階段訓練(18–40 s)。

三拍:(a) 架構圖 (b) 兩階段訓練 (c) 移除單一元件的代價。
論文把增益歸於注入結構與兩階段排程,本段即為此主張。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from style import (W, H, FPS, PANEL, canvas, draw_title, draw_subtitle, draw_note,
                   white_card, save_seq, ease, fig_to_image, dark_axes, font, COLORS)

ROOT = os.path.expanduser('~/SAM_research-1')
OUT = f'{ROOT}/docs/video/segments/s2'

SUB_A = 'A frozen SAM backbone, adapted through injection modules that start as the identity.'
SUB_C = 'Removing the two-stage schedule costs more than removing anything else.'

# root.tex tab:ablation,component removal 區塊
REMOVALS = [
    ('Without the injection adapter', -4.08, COLORS['CMA']),
    ('Without source-domain pre-training', -6.02, COLORS['Pair-SAM']),
]
SEED_STD = 0.14
FULL = 76.02

BEAT_A, BEAT_B, BEAT_C = 240, 195, 225        # 8.0s / 6.5s / 7.5s
FADE = 12                                      # 拍與拍之間的淡入淡出幀數


def card_frame(img_path, title, sub, note):
    c = canvas()
    d = ImageDraw.Draw(c, 'RGBA')
    card = white_card(Image.open(img_path), 1200, 460)
    c.paste(card, (40, 92))
    draw_title(d, title)
    draw_subtitle(d, sub)
    draw_note(d, note)
    return c


def removal_plot(progress):
    """水平長條由 0 向左生長,progress 為 0→1。"""
    fig, ax = plt.subplots(figsize=(10.4, 3.5), dpi=100)
    dark_axes(ax, fig)
    ax.grid(color='#2a2a2a', lw=0.6, axis='x')
    ys = np.arange(len(REMOVALS))[::-1]
    for y, (label, delta, col) in zip(ys, REMOVALS):
        v = delta * ease(progress)
        ax.barh(y, v, height=0.42, color=col)
        ax.text(-0.30, y, label, ha='right', va='center', color='#f2f2f2', fontsize=14)
        if progress > 0.55:
            a = min(1.0, (progress - 0.55) / 0.3)
            ax.text(v - 0.18, y, f'{delta:+.2f}', ha='right', va='center',
                    color=col, fontsize=15, weight='bold', alpha=a)
    ax.axvline(0, color='#777777', lw=1.0)
    ax.set_xlim(-7.2, 1.2)
    ax.set_ylim(-0.62, len(REMOVALS) - 0.38)
    ax.set_yticks([])
    ax.set_xlabel('change in mIoU when the component is removed (ACDC val)',
                  color='#aaaaaa', fontsize=11)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.93, bottom=0.16)
    img = fig_to_image(fig)
    plt.close(fig)
    return img


def beat_c_frame(progress):
    c = canvas()
    d = ImageDraw.Draw(c, 'RGBA')
    c.paste(removal_plot(progress), (60, 130))
    draw_title(d, 'Ablation  ·  ACDC validation split  ·  full model 76.02 mIoU')
    draw_subtitle(d, SUB_C)
    draw_note(d, 'Both removals are far larger than the 0.14 seed deviation of the full model.')
    return c


def main():
    frames = []
    a = card_frame(f'{ROOT}/assets/pairsam_overview.png',
                   'Architecture  ·  SAM ViT-H frozen, reference injected at four depths',
                   SUB_A,
                   'Zero-initialized per-channel gates: injection starts as the identity and grows only as the data warrant.')
    b = card_frame(f'{ROOT}/assets/training_stages.png',
                   'Two-stage schedule  ·  semantics on Cityscapes, then weather on the target domain',
                   SUB_A,
                   'Stage 1 learns semantic decoding; stage 2 spends its budget almost entirely on adaptation.')
    frames += [a] * BEAT_A

    for i in range(FADE):                                   # a → b
        frames.append(Image.blend(a, b, ease((i + 1) / FADE)))
    frames += [b] * (BEAT_B - FADE)

    c_first = beat_c_frame(0.0)
    for i in range(FADE):                                   # b → c
        frames.append(Image.blend(b, c_first, ease((i + 1) / FADE)))

    grow = 90                                               # 3s 生長
    for i in range(BEAT_C - FADE):
        frames.append(beat_c_frame(min(1.0, i / grow)) if i <= grow else frames[-1])

    save_seq(frames, OUT)
    print(f'S2: {len(frames)} frames ({len(frames)/FPS:.2f}s) -> {OUT}')


if __name__ == '__main__':
    main()
