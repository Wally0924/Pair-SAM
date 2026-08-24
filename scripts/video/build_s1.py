"""S1 — 夜間駕駛使既有模型崩潰(0–18 s)。

Dark Zurich GOPR0356 連續 50 幀,雙軌:Input / SegFormer(source-only)。
建立問題,不做方法比較。
"""
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from style import (W, H, FPS, canvas, draw_title, draw_subtitle, draw_note,
                   draw_label, fit, save_seq, COLORS)

DZ = os.path.expanduser('~/Datasets/Dark_Zurich/Dark_Zurich_val_anon')
SEG = os.path.expanduser('~/PairSAM_weights/06_baselines/cma/qual_results/'
                         'segformer_source/DarkZurich/color')
OUT = os.path.expanduser('~/SAM_research-1/docs/video/segments/s1')

CELL_W, CELL_H = 624, 351
GX, GY = 12, 128
HOLD = 11                                  # 30fps / 11 ≈ 2.7 fps 逐幀推進

SUB = 'A segmentation model trained on clear weather collapses at night.'


def compose(name):
    c = canvas()
    d = ImageDraw.Draw(c, 'RGBA')
    stem = name[:-4]

    inp = Image.open(f'{DZ}/rgb_anon/val/night/GOPR0356/{name}')
    pred = Image.open(f'{SEG}/{name}')
    for i, (img, label, col) in enumerate([
            (inp, 'Night input', '#ffffff'),
            (pred, 'SegFormer  (trained on clear weather only)', COLORS['SegFormer'])]):
        x = GX + i * (CELL_W + 8)
        c.paste(fit(img, CELL_W, CELL_H), (x, GY))
        d.rectangle([x, GY, x + CELL_W - 1, GY + CELL_H - 1], outline='#333333')
        draw_label(d, x + 6, GY + 6, label, col)

    draw_title(d, 'Dark Zurich  ·  night sequence GOPR0356  ·  50 consecutive frames')
    draw_subtitle(d, SUB)
    draw_note(d, 'SegFormer is the unadapted source model: trained on Cityscapes, never adapted to night.')
    return c


def main():
    names = sorted(os.listdir(f'{DZ}/rgb_anon/val/night/GOPR0356'))
    frames = []
    for n in names:
        fr = compose(n)
        frames += [fr] * HOLD
    save_seq(frames, OUT)
    print(f'S1: {len(frames)} frames ({len(frames)/FPS:.2f}s) -> {OUT}')


if __name__ == '__main__':
    main()
