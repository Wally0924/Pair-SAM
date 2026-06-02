# segment-anything/scripts/precompute_class_presence.py
"""
掃描訓練集 GT 遮罩，建立「每影像含哪些類別」表與全域類別像素計數，供 RareClassSampler 使用。

用法：
  python scripts/precompute_class_presence.py \
    --csv /home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_train.csv \
    --out /home/rvl1421/SAM_research-1/Datasets/class_presence.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def scan_gt_mask(gt_path, num_classes=19):
    """回傳 (present_classes:list[int], counts:list[int] len=num_classes)。255/越界忽略。"""
    m = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"無法讀取 GT: {gt_path}")
    counts = [0] * num_classes
    vals, cnts = np.unique(m, return_counts=True)
    for v, c in zip(vals.tolist(), cnts.tolist()):
        if 0 <= v < num_classes:
            counts[v] = int(c)
    present = [c for c in range(num_classes) if counts[c] > 0]
    return present, counts


def build_class_presence(csv_path, out_path, num_classes=19):
    df = pd.read_csv(csv_path)
    if 'gt_path' not in df.columns:
        raise ValueError("CSV 缺少 gt_path 欄位")
    presence = {}
    total_counts = [0] * num_classes
    for gt in tqdm(df['gt_path'].tolist(), desc='scan GT'):
        present, counts = scan_gt_mask(gt, num_classes)
        presence[gt] = present
        for c in range(num_classes):
            total_counts[c] += counts[c]
    data = {
        'num_classes': num_classes,
        'presence': presence,
        'class_pixel_counts': {c: total_counts[c] for c in range(num_classes)},
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_classes', type=int, default=19)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if os.path.isfile(args.out) and not args.force \
            and os.path.getmtime(args.out) >= os.path.getmtime(args.csv):
        print(f"✅ 快取已是最新，略過：{args.out}（--force 可強制重建）")
        return
    data = build_class_presence(args.csv, args.out, args.num_classes)
    n = len(data['presence'])
    print(f"✅ 掃描 {n} 張 GT → {args.out}")
    print(f"   class_pixel_counts: {data['class_pixel_counts']}")


if __name__ == '__main__':
    main()
