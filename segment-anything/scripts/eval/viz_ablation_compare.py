#!/usr/bin/env python
"""消融定性比較圖：每個天氣條件一列，欄位 = Input / GT / R1 / A2 / FULL。

config-aware：每個變體依其 ablation_config.json 重建（decoder/lrh/ref 各異），
推論流程與 eval_e1_acdc_val_full.py 完全一致（assemble_semantic_logits + use_lrh），
故定性預測與報告中的 mIoU 同源。

VRAM 安全：一次只載入一個變體（跑完 4 條件後釋放再載下一個），單卡 24GB 可行。
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.eval._eval_common import (
    load_pair_sam_from_ablation, make_batched_input, pick_first_per_condition,
    colorize_19class, denorm_image, CONDITION_NAMES, DEFAULT_VAL_CSV,
)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits
from utils.pair_dataloader import PairSegmentationDataset

OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation')
SAVE = os.path.join(OUT_ROOT, 'fig_ablation_qualitative.png')
DEVICE = 'cuda'
NUM_CLASSES = 19

# 要比較的變體（欄位順序）：標籤 → run 目錄
VARIANTS = [
    ('R1 (baseline)', 'R1_seed42'),
    ('A2 (no-ref)',   'A2_seed42'),
    ('FULL',          'R7_seed42'),
]


@torch.no_grad()
def infer_one(model, item):
    """單張推論 → (H,W) argmax 預測（與 eval_e1 流程一致）。"""
    batch = {
        'image':         item['image'].unsqueeze(0),
        'clear_image':   item['clear_image'].unsqueeze(0),
        'gt_mask':       item['gt_mask'].unsqueeze(0),
        'invalid_mask':  item['invalid_mask'].unsqueeze(0),
        'text_prompts':  [item['text_prompts']],
        'original_size': [item['original_size']],
        'condition_id':  item['condition_id'].unsqueeze(0),
    }
    outputs = model(make_batched_input(batch, DEVICE))
    low_res = outputs[0]['low_res_logits'].squeeze(0)
    class_ids_out = outputs[0]['class_ids']
    fused = assemble_semantic_logits(
        low_res, class_ids_out,
        fusion_head=model.context_fusion_head,
        num_classes=NUM_CLASSES,
        use_lrh=getattr(model, 'use_lrh', True),
    )
    fused_hr = F.interpolate(fused, size=(1024, 1024), mode='bilinear', align_corners=False)
    return fused_hr.argmax(dim=1).squeeze(0).cpu().numpy()


def main():
    ds = PairSegmentationDataset(
        csv_file=DEFAULT_VAL_CSV, image_size=1024, mode='val', force_raw_images=True,
    )
    picked = pick_first_per_condition(DEFAULT_VAL_CSV)
    conds = list(CONDITION_NAMES.keys())  # [0,1,2,3]
    print('Picked indices:', picked)

    # 預載每個條件的 item（input/GT 只需一份）
    items = {cid: ds[picked[cid]] for cid in conds}

    # preds[(variant_label, cid)] = colorized array；一次只載一個模型
    preds = {}
    for label, run_dir in VARIANTS:
        ckpt = os.path.join(OUT_ROOT, run_dir, 'weather_sam_best_latest.pth')
        cfg_path = os.path.join(OUT_ROOT, run_dir, 'ablation_config.json')
        print(f'\n[load] {label} ← {run_dir}')
        model, cfg = load_pair_sam_from_ablation(ckpt, cfg_path, device=DEVICE)
        model.eval()
        print(f'       cfg = {cfg}')
        for cid in conds:
            pred = infer_one(model, items[cid])
            preds[(label, cid)] = colorize_19class(pred)
        del model
        torch.cuda.empty_cache()

    # 繪圖：行=4 條件，欄= Input / GT / 各變體
    col_titles = ['Input (Adverse)', 'Ground Truth'] + [lab for lab, _ in VARIANTS]
    ncol = len(col_titles)
    fig, axes = plt.subplots(len(conds), ncol, figsize=(3.2 * ncol, 3.2 * len(conds)))

    for row, cid in enumerate(conds):
        item = items[cid]
        gt_np = item['gt_mask'].cpu().numpy().copy()
        gt_np[item['invalid_mask'].cpu().numpy().astype(bool)] = 255

        axes[row, 0].imshow(denorm_image(item['image']))
        axes[row, 1].imshow(colorize_19class(gt_np))
        for j, (label, _) in enumerate(VARIANTS):
            axes[row, 2 + j].imshow(preds[(label, cid)])

        axes[row, 0].set_ylabel(CONDITION_NAMES[cid].capitalize(),
                                fontsize=13, fontweight='bold')
        for col in range(ncol):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(SAVE, dpi=150)
    print(f'\n✅ 定性比較圖 → {SAVE}')


if __name__ == '__main__':
    main()
