# segment-anything/scripts/eval/viz_e4_qualitative.py
"""
E4: 定性比較圖 — 4 條件各 1 張樣本 × 3 欄（input / our pred / GT）
對應論文：Refign Fig.4, CMA Fig.4（缺 baseline 欄，留待 E2 head-to-head 完成後補）
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (
    load_v15_model, make_batched_input, pick_first_per_condition, denorm_image,
    CONDITION_NAMES, OUTPUT_ROOT, DEFAULT_CKPT, DEFAULT_VAL_CSV,
    colorize_19class,
)

NUM_CLASSES = 19
DEVICE = 'cuda'


def main():
    model = load_v15_model(DEFAULT_CKPT, device=DEVICE)

    # 為了取「指定 index 的 sample」，直接使用 Dataset 而非 DataLoader
    from utils.weather_dataloader import WeatherSegmentationDataset
    ds = WeatherSegmentationDataset(
        csv_file=DEFAULT_VAL_CSV, image_size=1024, mode='val', force_raw_images=True,
    )
    picked = pick_first_per_condition(DEFAULT_VAL_CSV)
    print('Picked indices:', picked)

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 18))
    column_titles = ['Input (Adverse)', 'WeatherSAM (Ours)', 'Ground Truth']

    with torch.no_grad():
        for row, cid in enumerate(CONDITION_NAMES):
            idx = picked[cid]
            item = ds[idx]
            # 模擬 collate_fn 但 batch_size=1
            batch = {
                'image':         item['image'].unsqueeze(0),
                'clear_image':   item['clear_image'].unsqueeze(0),
                'gt_mask':       item['gt_mask'].unsqueeze(0),
                'invalid_mask':  item['invalid_mask'].unsqueeze(0),
                'text_prompts':  [item['text_prompts']],
                'original_size': [item['original_size']],
                'condition_id':  item['condition_id'].unsqueeze(0),
            }
            batched_input = make_batched_input(batch, DEVICE)
            outputs = model(batched_input)

            low_res = outputs[0]['low_res_logits'].squeeze(0)
            class_ids_out = outputs[0]['class_ids']
            full = torch.full(
                (1, NUM_CLASSES, 256, 256), -10.0,
                device=DEVICE, dtype=low_res.dtype,
            )
            for k, c in enumerate(class_ids_out):
                full[0, c] = low_res[k]
            fused = model.context_fusion_head(full)
            fused_hr = F.interpolate(
                fused, size=(1024, 1024), mode='bilinear', align_corners=False,
            )
            pred = fused_hr.argmax(dim=1).squeeze(0).cpu().numpy()

            # 把 GT 中 invalid 區塊設為 255（顯示為黑色，與 colorize 一致）
            gt_np = item['gt_mask'].cpu().numpy().copy()
            invalid_np = item['invalid_mask'].cpu().numpy().astype(bool)
            gt_np[invalid_np] = 255

            # 渲染 3 欄
            axes[row, 0].imshow(denorm_image(item['image']))
            axes[row, 1].imshow(colorize_19class(pred))
            axes[row, 2].imshow(colorize_19class(gt_np))

            # row label（左側）
            axes[row, 0].set_ylabel(
                CONDITION_NAMES[cid].capitalize(),
                fontsize=14, fontweight='bold',
            )
            for col in range(3):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

    # column titles（最上方一列）
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / 'e4_qualitative.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Figure written: {out_path}')


if __name__ == '__main__':
    main()
