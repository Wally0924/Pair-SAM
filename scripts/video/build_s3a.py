"""S3a — 三條件廣度(40–52 s)。

fog / rain / snow 三格橫排,每格由 Input 疊化到 Pair-SAM 預測。
素材為論文 Fig. 的同一批檔案(FULL_seed42),不使用 6/21–6/28 的舊產物。
"""
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from style import (W, H, FPS, canvas, draw_title, draw_subtitle, draw_note,
                   draw_label, fit, save_seq, ease, font, COLORS)

Q = os.path.expanduser('~/SAM_research-1/paper/conference_paper/Images/qual/ACDC')
OUT = os.path.expanduser('~/SAM_research-1/docs/video/segments/s3a')

ROWS = [('row1', 'fog'), ('row2', 'rain'), ('row3', 'snow')]
CELL_W, CELL_H = 405, 304          # 中央 4:3 裁切,避免三格橫排過扁
GX, GY, GAP = 20, 176, 12

HOLD_IN, XFADE, HOLD_OUT = 105, 60, 195       # 3.5s 輸入 / 2s 疊化 / 6.5s 預測

SUB = 'The same frozen backbone adapts across fog, rain, and snow.'


def crop43(img):
    """取中央 4:3 區域,保留路面與兩側結構。"""
    w, h = img.size
    tw = int(h * 4 / 3)
    x0 = (w - tw) // 2
    return img.crop((x0, 0, x0 + tw, h))


def compose(t):
    """t: 0→1 的疊化進度。"""
    c = canvas()
    d = ImageDraw.Draw(c, 'RGBA')
    for i, (row, cond) in enumerate(ROWS):
        inp = fit(crop43(Image.open(f'{Q}/{row}_{cond}_col1_Input.png')), CELL_W, CELL_H)
        out = fit(crop43(Image.open(
            f'{Q}/Ours_Pair-SAM_FULLseed42/{row}_{cond}_col7_Pair-SAM.png')), CELL_W, CELL_H)
        cell = inp if t <= 0 else (out if t >= 1 else Image.blend(inp, out, ease(t)))
        x = GX + i * (CELL_W + GAP)
        c.paste(cell, (x, GY))
        d.rectangle([x, GY, x + CELL_W - 1, GY + CELL_H - 1], outline='#333333')
        draw_label(d, x + 6, GY + 6, cond.upper(), '#ffffff', 14)
        tag = 'input' if t < 0.5 else 'Pair-SAM'
        col = '#bbbbbb' if t < 0.5 else COLORS['Pair-SAM']
        draw_label(d, x + CELL_W - 96, GY + CELL_H - 28, tag, col, 12)

    stage = 'adverse input' if t < 0.5 else 'Pair-SAM prediction'
    f = font(20, True)
    d.text(((W - d.textlength(stage, font=f)) / 2, 118), stage,
           fill='#dddddd' if t < 0.5 else COLORS['Pair-SAM'], font=f)
    draw_title(d, 'ACDC validation  ·  one frame per condition  ·  same checkpoint throughout')
    draw_subtitle(d, SUB)
    draw_note(d, 'Frames and predictions are the ones printed in the paper.')
    return c


def main():
    frames = []
    frames += [compose(0.0)] * HOLD_IN
    for i in range(XFADE):
        frames.append(compose((i + 1) / XFADE))
    frames += [compose(1.0)] * HOLD_OUT
    save_seq(frames, OUT)
    print(f'S3a: {len(frames)} frames ({len(frames)/FPS:.2f}s) -> {OUT}')


if __name__ == '__main__':
    main()
