"""MUSES 驗證集定性視覺化:每個樣本輸出一張含四格的合成圖。

四格依序為「輸入影像 | 參考影像 | 標註 | 本方法預測」,即一個自含式展示;
4 個樣本 = 4 張獨立合成圖,供論文以 subfigure 自由排版。另輸出一張共用的
19 類色標(legend)PNG。固定使用 FULL_seed42 權重、cond off(text + reference)。

預設避開 clear-day(其參考影像即自身,會與輸入重複),改選有真實晴天參考的天氣。

用法
----
    conda run -n sam_env python scripts/eval/viz_muses_qual.py \
        --buckets rain:day snow:day rain:night snow:night \
        --out-dir ../paper/Chinese_master_thesis/Images

每個 bucket 取該 (天氣, 光照) 於 val CSV 的第一筆(穩定可重現)。輸出:
    ch4_muses_qual_<tag>.png        （四格合成,<tag>=day1/day2/night1/night2）
    ch4_muses_qual_legend.png
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
    load_pair_sam_from_ablation, make_batched_input,
    colorize_19class, CITYSCAPES_CLASSES, CITYSCAPES_PALETTE,
)
from dump_muses_preds import predict_native, resolve_condition_csv  # noqa: E402

_SEGANY_ROOT = _THIS.parents[2]
if str(_SEGANY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGANY_ROOT))
from utils.pair_dataloader import PairSegmentationDataset  # noqa: E402

DEFAULT_CKPT = str(_SEGANY_ROOT / 'outputs_ablation_m2f' / 'FULL_seed42'
                   / 'weather_sam_best_latest.pth')


def save_composite(input_rgb: np.ndarray, ref_rgb: np.ndarray,
                   gt_color: np.ndarray, pred_color: np.ndarray,
                   title: str, out_path: Path) -> None:
    """把四格「輸入 | 參考 | 標註 | 預測」拼成一張帶標題的合成圖。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [(input_rgb, 'Input'), (ref_rgb, 'Reference'),
              (gt_color, 'Ground Truth'), (pred_color, 'Result (Ours)')]
    fig, axes = plt.subplots(1, 4, figsize=(22, 3.4))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    for ax, (img, cap) in zip(axes, panels):
        ax.imshow(img); ax.set_title(cap, fontsize=15); ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def make_legend(out_path: Path, ncol: int = 5) -> None:
    """輸出 19 類色標 PNG(色塊 + 類別名),供論文共用一張。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=CITYSCAPES_PALETTE[i] / 255.0,
                              label=CITYSCAPES_CLASSES[i]) for i in range(19)]
    fig = plt.figure(figsize=(12, 1.4))
    fig.legend(handles=handles, loc='center', ncol=ncol, frameon=False, fontsize=11)
    plt.axis('off')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = _SEGANY_ROOT.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--csv', type=str,
                   default=str(repo_root / 'Datasets' / 'muses_ref_rgb_val.csv'))
    p.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    p.add_argument('--buckets', nargs='+',
                   default=['rain:day', 'snow:day', 'rain:night', 'snow:night'],
                   help='每個 bucket 形如 weather:tod,取該組於 val 的第一筆')
    p.add_argument('--out-dir', type=str,
                   default=str(repo_root / 'paper' / 'Chinese_master_thesis' / 'Images'))
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, _cfg = load_pair_sam_from_ablation(args.ckpt, device=args.device)
    model.use_cond = False  # cond off:text + reference 照常

    resolved = resolve_condition_csv(args.csv, 'off')
    ds = PairSegmentationDataset(csv_file=resolved, image_size=1024,
                                    mode='val', force_raw_images=True)
    df = ds.data.reset_index(drop=True)

    day_n = night_n = 0
    summary = []
    for spec in args.buckets:
        weather, tod = spec.split(':')
        cand = df.index[(df['weather'] == weather) & (df['time_of_day'] == tod)].tolist()
        if not cand:
            print(f'⚠️  無 {spec} 樣本,跳過'); continue
        idx = cand[0]
        row = df.iloc[idx]
        if tod == 'day':
            day_n += 1; tag = f'day{day_n}'
        else:
            night_n += 1; tag = f'night{night_n}'

        # 前向(原生解析度)
        item = PairSegmentationDataset.collate_fn([ds[idx]])
        batched = make_batched_input(item, args.device)
        gt = cv2.imread(str(row['gt_path']), cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape[:2]
        with torch.no_grad():
            pred = predict_native(model, item, args.device, target_hw=(H, W))

        # 四格合成:輸入 | 參考 | 標註 | 預測(皆原生 1920x1080)
        inp = cv2.cvtColor(cv2.imread(str(row['image_path'])), cv2.COLOR_BGR2RGB)
        ref = cv2.cvtColor(cv2.imread(str(row['ref_image_path'])), cv2.COLOR_BGR2RGB)
        title = f'{weather.capitalize()} / {tod.capitalize()}'
        out_path = out_dir / f'ch4_muses_qual_{tag}.png'
        save_composite(inp, ref, colorize_19class(gt.astype(np.uint8)),
                       colorize_19class(pred.astype(np.uint8)), title, out_path)
        summary.append((tag, weather, tod, row['image_path']))
        print(f'✅ {tag}: {weather}/{tod}  ← {Path(row["image_path"]).name}')

    make_legend(out_dir / 'ch4_muses_qual_legend.png')
    print(f'\n📁 輸出目錄:{out_dir}')
    print('對照表(tag → 天氣/光照):')
    for tag, w, t, p in summary:
        print(f'  {tag}: {w} / {t}')


if __name__ == '__main__':
    main()
