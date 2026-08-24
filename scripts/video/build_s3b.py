"""S3b — 零樣本 vs 目標域適應（48–78 s）。

左側 2x2 四軌逐幀播放 Dark Zurich GOPR0356 全 50 幀，右側同步推進逐幀 mIoU 折線。
輸出 1280x720 @30fps 的影格序列，交由 ffmpeg 編碼。
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from PIL import Image, ImageDraw

F = os.path.expanduser('~/Downloads/figures')
B = os.path.expanduser('~/PairSAM_weights/06_baselines')
DZ = os.path.expanduser('~/Datasets/Dark_Zurich/Dark_Zurich_val_anon')
GT = f'{DZ}/gt/val/night/GOPR0356'
OUT = os.path.expanduser('~/SAM_research-1/docs/video/segments/s3b')

W, H = 1280, 720
CELL_W, CELL_H = 448, 252          # 2x2 每格
GRID_X, GRID_Y = 12, 46
PLOT_W, PLOT_H = 352, 508
PLOT_X, PLOT_Y = 916, 46
HOLD = 12                           # 每個資料幀保持的影片幀數（30fps → 0.4s）
TAIL_S = 10                         # 段末定格秒數

TRACKS = [
    ('Night input', None),
    ('CMA  (target-adapted)', f'{B}/cma/qual_results/cma_segformer/DarkZurich/color'),
    ('Refign  (target-adapted)', f'{B}/refign/qual_results/DarkZurich/color'),
    ('Pair-SAM  (zero-shot)', f'{F}/_ours_color_dz_paper'),
]
LINE_COLORS = {'CMA': '#e8833a', 'Refign': '#4f9dd9', 'Pair-SAM': '#d94f4f'}


sys.path.insert(0, os.path.dirname(__file__))
from style import (font, canvas as _canvas, draw_title, draw_subtitle,  # noqa: E402
                   draw_note, draw_label)


def _cm(pred, gt, invalid):
    """單張混淆矩陣，排除 GT ignore 與 Dark Zurich 的 invalid 區域。"""
    m = (gt != 255) & (invalid == 0)
    return np.bincount(gt[m].astype(np.int64) * 19 + pred[m].astype(np.int64),
                       minlength=361).reshape(19, 19)


def _miou_from_cm(cm):
    """論文口徑:資料集層級 mIoU,只計 GT 中出現過的類別(DZ val 為 17 類)。"""
    inter = np.diag(cm).astype(float)
    union = cm.sum(1) + cm.sum(0) - inter
    present = cm.sum(1) > 0
    return float(np.mean(inter[present] / union[present]) * 100)


def running_miou(names):
    """逐幀累積的資料集層級 mIoU,終點即論文 Table「Generalization」的數字。"""
    srcs = {
        'CMA': f'{B}/cma/qual_results/cma_segformer/DarkZurich/trainId',
        'Refign': f'{B}/refign/qual_results/DarkZurich/trainId',
        'Pair-SAM': f'{F}/_ours_pred_dz_paper',
    }
    out = {}
    for k, d in srcs.items():
        cm = np.zeros((19, 19), np.int64)
        vals = []
        for n in names:
            base = n[:-4]
            stem = base.replace('_rgb_anon', '')
            g = np.array(Image.open(f'{GT}/{stem}_gt_labelTrainIds.png'))
            inv = np.array(Image.open(f'{GT}/{stem}_gt_invIds.png'))
            cm += _cm(np.array(Image.open(f'{d}/{base}.png')), g, inv)
            vals.append(_miou_from_cm(cm))
        out[k] = np.array(vals)
    return out


def render_plot(curves, upto, n):
    """畫到第 upto 幀（含）為止的折線，回傳 PIL Image。"""
    fig, ax = plt.subplots(figsize=(PLOT_W / 100, PLOT_H / 100), dpi=100)
    fig.patch.set_facecolor('#141414')
    ax.set_facecolor('#141414')
    x = np.arange(1, n + 1)
    for k, v in curves.items():
        lw = 3.0 if k == 'Pair-SAM' else 1.6
        ax.plot(x[:upto + 1], v[:upto + 1], color=LINE_COLORS[k], lw=lw, label=k)
        ax.scatter([x[upto]], [v[upto]], color=LINE_COLORS[k], s=34 if k == 'Pair-SAM' else 16, zorder=3)
        if upto == len(v) - 1:
            ax.annotate(f'{v[-1]:.1f}', (x[-1], v[-1]), textcoords='offset points',
                        xytext=(-30, 6), color=LINE_COLORS[k], fontsize=9, weight='bold')
    ax.axvline(x[upto], color='#666666', lw=0.8, ls=':')
    ax.set_xlim(1, n)
    ax.set_ylim(30, 62)
    ax.set_xlabel('frame', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('mIoU (%)', color='#aaaaaa', fontsize=9)
    ax.set_title('mIoU accumulated over the sequence', color='#8a8a8a', fontsize=8, pad=6)
    ax.tick_params(colors='#888888', labelsize=8)
    for s in ax.spines.values():
        s.set_color('#3a3a3a')
    ax.grid(color='#2a2a2a', lw=0.6)
    leg = ax.legend(loc='lower left', fontsize=8, facecolor='#1e1e1e', edgecolor='#3a3a3a', framealpha=0.9)
    for t in leg.get_texts():
        t.set_color('#dddddd')
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1)
    plt.close(fig)
    return img.convert('RGB')


def compose(idx, names, curves, tail=False):
    canvas = _canvas()
    d = ImageDraw.Draw(canvas, 'RGBA')
    n = names[idx]
    base = n[:-4]

    for i, (label, src) in enumerate(TRACKS):
        path = (f'{DZ}/rgb_anon/val/night/GOPR0356/{base}.png' if src is None
                else f'{src}/{base}.png')
        cell = Image.open(path).convert('RGB').resize((CELL_W, CELL_H), Image.BILINEAR)
        x = GRID_X + (i % 2) * (CELL_W + 4)
        y = GRID_Y + (i // 2) * (CELL_H + 4)
        canvas.paste(cell, (x, y))
        d.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], outline='#333333')
        tag = label.split('  ')[0]
        draw_label(d, x + 6, y + 6, label, LINE_COLORS.get(tag, '#ffffff'), 13)

    canvas.paste(render_plot(curves, idx, len(names)), (PLOT_X, PLOT_Y))

    draw_title(d, 'Dark Zurich  ·  night sequence GOPR0356  ·  50 consecutive frames')

    sub = ('CMA and Refign were adapted on this target domain. Pair-SAM never saw it.'
           if idx < len(names) // 2 else
           'Zero-shot, it matches or exceeds them on most frames.')
    if tail:
        sub = '54.1 zero-shot, against 50.0 and 48.9 with target adaptation.'
    draw_subtitle(d, sub)
    draw_note(d, 'mIoU over the 17 classes present in Dark Zurich val, invalid regions '
                 'excluded. All 50 frames shown, none excluded.')
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(x for x in os.listdir(f'{F}/_ours_pred_dz_paper')
                   if x.endswith('.png') and not x.startswith('_'))
    curves = running_miou(names)
    print('final aggregate mIoU:', {k: round(v[-1], 2) for k, v in curves.items()},
          '  (paper: Ours 54.2 / CMA 50.0 / Refign 48.9)')

    k = 0
    for i in range(len(names)):
        frame = compose(i, names, curves)
        for _ in range(HOLD):
            frame.save(f'{OUT}/{k:05d}.png'); k += 1
    tail = compose(len(names) - 1, names, curves, tail=True)
    for _ in range(TAIL_S * 30):
        tail.save(f'{OUT}/{k:05d}.png'); k += 1
    print(f'{k} video frames ({k / 30:.1f}s) -> {OUT}')


if __name__ == '__main__':
    main()
