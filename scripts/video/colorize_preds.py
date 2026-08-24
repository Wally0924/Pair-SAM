"""把 Pair-SAM 匯出的 trainId 預測（.npy）著色成 RGB PNG。

輸出檔名與 baseline 的 qual_results/*/color/*.png 對齊，方便逐幀併軌。

用法：
    python scripts/video/colorize_preds.py \
        --src ~/Downloads/figures/_ours_pred_darkzurich \
        --dst ~/Downloads/figures/_ours_color_darkzurich
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'segment-anything'))
from scripts.eval._eval_common import colorize_19class  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='含 .npy 預測的目錄')
    ap.add_argument('--dst', required=True, help='輸出 PNG 目錄')
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)
    os.makedirs(dst, exist_ok=True)

    # 底線開頭的檔案是統計副產物（例如 _per_frame_cm.npy），不是預測
    names = sorted(n for n in os.listdir(src)
                   if n.endswith(('.npy', '.png')) and not n.startswith('_'))
    if not names:
        raise SystemExit(f'{src} 內找不到 .npy / .png 預測')

    for name in names:
        stem, ext = os.path.splitext(name)
        mask = (np.load(os.path.join(src, name)) if ext == '.npy'
                else np.array(Image.open(os.path.join(src, name))))
        out = os.path.join(dst, stem + '.png')
        Image.fromarray(colorize_19class(mask)).save(out)

    print(f'{len(names)} frames -> {dst}')


if __name__ == '__main__':
    main()
