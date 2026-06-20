"""通用 val 定性視覺化：input | 預測 | GT，適用任意 val CSV（ACDC / Dark Zurich…）。
挑 N 張（在整個 val 上均勻取樣）以 config 正確重建模型推論。純推論。

用法：
    conda run -n sam_env python scripts/eval/viz_val_samples.py \
        --ckpt outputs_ablation/R7_seed42/weather_sam_best_latest.pth \
        --csv ../Datasets/darkzurich_adverse_ref_rgb_val.csv \
        --out /home/rvl1421/Downloads/figures/dzval_qualitative.png --n 6
"""
import argparse, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[1]))
from _eval_common import (load_weather_sam_from_ablation, denorm_image, colorize_19class)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits
from utils.weather_dataloader import WeatherSegmentationDataset

NUM_CLASSES = 19
DEVICE = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6, help='視覺化張數')
    args = ap.parse_args()

    model, cfg = load_weather_sam_from_ablation(args.ckpt, args.config, device=DEVICE)
    print(f'[viz] config: {cfg}')
    ds = WeatherSegmentationDataset(csv_file=args.csv, image_size=1024,
                                    mode='val', force_raw_images=True)
    n = min(args.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).round().astype(int).tolist()  # 均勻取樣
    print(f'samples: {len(ds)}  picked idx: {idxs}')

    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.5 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ['Input (Adverse)', 'WeatherSAM (Ours)', 'Ground Truth']

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            item = ds[idx]
            batch = {k: item[k].unsqueeze(0) for k in
                     ['image', 'clear_image', 'gt_mask', 'invalid_mask', 'condition_id']}
            batch['text_prompts'] = [item['text_prompts']]
            batch['original_size'] = [item['original_size']]
            bi = [{
                'image': batch['image'][0].to(DEVICE),
                'clear_image': batch['clear_image'][0].to(DEVICE),
                'text_prompts': batch['text_prompts'][0],
                'original_size': batch['original_size'][0],
                'condition_id': batch['condition_id'][0],
            }]
            outputs = model(bi)
            low_res = outputs[0]['low_res_logits'].squeeze(0)
            class_ids_out = outputs[0]['class_ids']
            fused = assemble_semantic_logits(
                low_res, class_ids_out, fusion_head=model.context_fusion_head,
                num_classes=NUM_CLASSES, use_lrh=getattr(model, 'use_lrh', True))
            fused_hr = F.interpolate(fused, size=(1024, 1024), mode='bilinear', align_corners=False)
            pred = fused_hr.argmax(dim=1).squeeze(0).cpu().numpy()

            gt_np = item['gt_mask'].cpu().numpy().copy()
            gt_np[item['invalid_mask'].cpu().numpy().astype(bool)] = 255

            axes[row, 0].imshow(denorm_image(item['image']))
            axes[row, 1].imshow(colorize_19class(pred))
            axes[row, 2].imshow(colorize_19class(gt_np))
            axes[row, 0].set_ylabel(f'#{idx}', fontsize=12, fontweight='bold')
            for col in range(3):
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Figure written: {args.out}')


if __name__ == '__main__':
    main()
