"""MUSES 定性視覺化：輸出單張切片，供論文以 subfigure 版型自由排版。

與 viz_muses_qual.py 的差異：後者輸出四格已拼好的合成圖（含 matplotlib 標題與
留白），版型固定；本腳本改輸出**未加標註的單張 PNG**，欄位切分與 ACDC / Dark
Zurich 的 Images/qual/<dataset>/rowN_colM_<type>.png 命名一致，讓 LaTeX 端能用
同一套 subfigure 版型排版。

MUSES 無 baseline 對照（SegFormer / Refign / CMA 未在此資料集上跑），因此每列
只有 4 欄：Input / Reference / GroundTruth / Ours。

用法
----
    conda run -n sam_env python scripts/eval/dump_muses_qual_slices.py \
        --buckets rain:day snow:day rain:night snow:night

每個 bucket 取該 (天氣, 光照) 於 val CSV 的第一筆（穩定可重現），輸出至
    ../paper/Chinese_master_thesis/Images/qual/MUSES/rowN_colM_<type>.png
並印出 rowN → 天氣/光照 的對照表，供 LaTeX caption 對應。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (  # noqa: E402
    load_pair_sam_from_ablation, colorize_19class,
)
from dump_muses_preds import predict_native, resolve_condition_csv  # noqa: E402

_SEGANY_ROOT = _THIS.parents[2]
if str(_SEGANY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGANY_ROOT))
from utils.pair_dataloader import PairSegmentationDataset  # noqa: E402

DEFAULT_CKPT = str(_SEGANY_ROOT / 'outputs_ablation_m2f' / 'FULL_seed42'
                   / 'weather_sam_best_latest.pth')

# 與 ACDC / DarkZurich 切片一致的輸出寬度；等比縮放，不改變長寬比
SLICE_WIDTH = 512


def save_slice(img_rgb: np.ndarray, out_path: Path) -> None:
    """等比縮放到固定寬度後存檔（RGB 進、BGR 出）。"""
    h, w = img_rgb.shape[:2]
    new_h = round(h * SLICE_WIDTH / w)
    # 分割色圖用 NEAREST 保邊界，照片用 AREA 縮圖較平滑
    interp = cv2.INTER_AREA
    resized = cv2.resize(img_rgb, (SLICE_WIDTH, new_h), interpolation=interp)
    cv2.imwrite(str(out_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))


def save_slice_label(mask: np.ndarray, out_path: Path) -> None:
    """分割標籤先縮放（NEAREST）再上色，避免色彩內插產生不存在的類別色。"""
    h, w = mask.shape[:2]
    new_h = round(h * SLICE_WIDTH / w)
    resized = cv2.resize(mask, (SLICE_WIDTH, new_h), interpolation=cv2.INTER_NEAREST)
    color = colorize_19class(resized.astype(np.uint8))
    cv2.imwrite(str(out_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    repo_root = _SEGANY_ROOT.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--csv', type=str,
                   default=str(repo_root / 'Datasets' / 'muses_ref_rgb_val.csv'))
    p.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    p.add_argument('--buckets', nargs='+',
                   default=['rain:day', 'snow:day', 'rain:night', 'snow:night'],
                   help='每個 bucket 形如 weather:tod，取該組於 val 的第一筆')
    p.add_argument('--out-dir', type=str,
                   default=str(repo_root / 'paper' / 'Chinese_master_thesis'
                               / 'Images' / 'qual' / 'MUSES'))
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, _cfg = load_pair_sam_from_ablation(args.ckpt, device=args.device)
    model.use_cond = False  # cond off：text + reference 照常

    resolved = resolve_condition_csv(args.csv, 'off')
    ds = PairSegmentationDataset(csv_file=resolved, image_size=1024,
                                 mode='val', force_raw_images=True)
    df = ds.data.reset_index(drop=True)

    summary = []
    for row_no, spec in enumerate(args.buckets, start=1):
        weather, tod = spec.split(':')
        cand = df.index[(df['weather'] == weather) & (df['time_of_day'] == tod)].tolist()
        if not cand:
            print(f'⚠️  無 {spec} 樣本，跳過'); continue
        idx = cand[0]
        row = df.iloc[idx]

        item = PairSegmentationDataset.collate_fn([ds[idx]])
        gt = cv2.imread(str(row['gt_path']), cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape[:2]
        with torch.no_grad():
            pred = predict_native(model, item, args.device, target_hw=(H, W))

        inp = cv2.cvtColor(cv2.imread(str(row['image_path'])), cv2.COLOR_BGR2RGB)
        ref = cv2.cvtColor(cv2.imread(str(row['ref_image_path'])), cv2.COLOR_BGR2RGB)

        save_slice(inp, out_dir / f'row{row_no}_col1_Input.png')
        save_slice(ref, out_dir / f'row{row_no}_col2_Reference.png')
        save_slice_label(gt, out_dir / f'row{row_no}_col3_GroundTruth.png')
        save_slice_label(pred, out_dir / f'row{row_no}_col4_Ours.png')

        summary.append((row_no, weather, tod, Path(row['image_path']).name))
        print(f'✅ row{row_no}: {weather}/{tod}  ← {Path(row["image_path"]).name}')

    print(f'\n📁 輸出目錄：{out_dir}')
    print('對照表（row → 天氣/光照）：')
    for row_no, w, t, name in summary:
        print(f'  row{row_no}: {w} / {t}  ({name})')


if __name__ == '__main__':
    main()
