# segment-anything/scripts/eval/eval_e1_acdc_val_full.py
"""
E1: ACDC val 完整評估
產出：per-class × per-condition IoU 矩陣 + 整體 mIoU + per-condition mIoU
對應論文：Refign Tab.1 / CMA Tab.1
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (
    load_weather_sam_model, load_weather_sam_from_ablation,
    build_acdc_val_loader, make_batched_input,
    CONDITION_NAMES, CITYSCAPES_CLASSES, OUTPUT_ROOT, DEFAULT_CKPT,
)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits

NUM_CLASSES = 19
IGNORE_INDEX = 255
DEVICE = 'cuda'


def iou_from_confusion(cm: torch.Tensor) -> torch.Tensor:
    """從 (C, C) 混淆矩陣計算 per-class IoU。空白類別回傳 NaN。"""
    tp    = cm.diag().float()
    fp    = cm.sum(dim=0).float() - tp
    fn    = cm.sum(dim=1).float() - tp
    denom = tp + fp + fn
    iou   = torch.where(denom > 0, tp / denom, torch.full_like(tp, float('nan')))
    return iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='該 run 的 checkpoint 路徑')
    ap.add_argument('--config', default=None, help='ablation_config.json（預設取 ckpt 同目錄）')
    ap.add_argument('--out', default=None, help='輸出 JSON 路徑（預設沿用原 OUTPUT_ROOT）')
    args = ap.parse_args()
    model, cfg = load_weather_sam_from_ablation(args.ckpt, args.config, device=DEVICE)
    print(f"[eval] loaded run config: {cfg}")
    loader = build_acdc_val_loader()

    # 5 個混淆矩陣：overall + 4 conditions
    cm_overall = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    cm_per_cond = {cid: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
                   for cid in CONDITION_NAMES.keys()}
    sample_counts = {cid: 0 for cid in CONDITION_NAMES.keys()}

    with torch.no_grad():
        for batch in tqdm(loader, desc='E1 ACDC val'):
            batched_input = make_batched_input(batch, DEVICE)
            outputs = model(batched_input)

            gt_mask = batch['gt_mask'][0].to(DEVICE).long()           # (H, W)
            invalid = batch['invalid_mask'][0].to(DEVICE)             # (H, W) bool
            cid     = int(batch['condition_id'][0].item())

            # 1. low_res_logits (K, 256, 256) → fused_logits_hr (1, 19, 1024, 1024)
            #    遵照 weather_trainer.validate_epoch 的流程
            low_res = outputs[0]['low_res_logits'].squeeze(0)         # (K, 256, 256)
            class_ids_out = outputs[0]['class_ids']                    # List[int]
            fused = assemble_semantic_logits(
                low_res, class_ids_out,
                fusion_head=model.context_fusion_head,
                num_classes=NUM_CLASSES,
                use_lrh=getattr(model, 'use_lrh', True),
            )
            fused_hr = F.interpolate(
                fused, size=(1024, 1024), mode='bilinear', align_corners=False,
            )

            pred = fused_hr.argmax(dim=1).squeeze(0)                   # (H, W)

            # 2. 過濾 ignore 像素（GT==255 或 invalid_mask）
            gt_used = gt_mask.clone()
            gt_used[invalid] = IGNORE_INDEX
            valid_px = gt_used != IGNORE_INDEX
            if not valid_px.any():
                continue

            g = gt_used[valid_px].cpu().long()
            p = pred[valid_px].cpu().long()

            # 3. 累積混淆矩陣
            cm_step = torch.bincount(
                g * NUM_CLASSES + p, minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            cm_overall = cm_overall + cm_step
            cm_per_cond[cid] = cm_per_cond[cid] + cm_step
            sample_counts[cid] += 1

    # ── 計算 IoU ──
    iou_overall = iou_from_confusion(cm_overall)
    iou_per_cond = {
        cid: iou_from_confusion(cm) for cid, cm in cm_per_cond.items()
    }

    def nanmean(t):
        return float(torch.nanmean(t).item()) if not torch.isnan(t).all() else float('nan')

    miou_overall = nanmean(iou_overall)
    miou_per_cond = {cid: nanmean(iou_per_cond[cid]) for cid in CONDITION_NAMES}

    # ── 輸出 JSON ──
    if args.out is not None:
        json_path = Path(args.out)
    else:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        json_path = OUTPUT_ROOT / 'e1_acdc_val_results.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        'checkpoint': str(Path(args.ckpt).name),
        'num_samples_total': sum(sample_counts.values()),
        'sample_counts_by_condition': {
            CONDITION_NAMES[cid]: n for cid, n in sample_counts.items()
        },
        'overall_miou': miou_overall,
        'per_condition_miou': {
            CONDITION_NAMES[cid]: miou_per_cond[cid] for cid in CONDITION_NAMES
        },
        'per_class_iou_overall': {
            CITYSCAPES_CLASSES[c]: (
                float(iou_overall[c]) if not math.isnan(float(iou_overall[c])) else None
            )
            for c in range(NUM_CLASSES)
        },
        'per_class_iou_by_condition': {
            CONDITION_NAMES[cid]: {
                CITYSCAPES_CLASSES[c]: (
                    float(iou_per_cond[cid][c])
                    if not math.isnan(float(iou_per_cond[cid][c])) else None
                )
                for c in range(NUM_CLASSES)
            }
            for cid in CONDITION_NAMES
        },
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'✅ JSON written: {json_path}')

    # ── 輸出 Markdown ──
    md_path = json_path.with_suffix('.md')
    lines = []
    lines.append('# E1: WeatherSAM v15 (E27) — ACDC val Evaluation')
    lines.append('')
    lines.append(f'**Checkpoint:** `{Path(args.ckpt).name}`')
    lines.append(f'**Date:** {datetime.now().strftime("%Y-%m-%d")}')
    lines.append(f'**Samples:** {sum(sample_counts.values())} ' +
                 '(' + ', '.join(f'{CONDITION_NAMES[cid]}={n}'
                                  for cid, n in sample_counts.items()) + ')')
    lines.append(f'**Overall mIoU:** {miou_overall*100:.2f}%')
    lines.append('')
    lines.append('## Per-Condition mIoU')
    lines.append('')
    lines.append('| Condition | mIoU (%) |')
    lines.append('|-----------|---------:|')
    for cid in CONDITION_NAMES:
        lines.append(f'| {CONDITION_NAMES[cid].capitalize():9s} | '
                     f'{miou_per_cond[cid]*100:.2f} |')
    lines.append(f'| **All**   | **{miou_overall*100:.2f}** |')
    lines.append('')
    lines.append('## Per-Class × Per-Condition IoU (%)')
    lines.append('')
    header = '| Class | Fog | Rain | Snow | Night | All |'
    sep    = '|-------|----:|-----:|-----:|------:|----:|'
    lines.append(header)
    lines.append(sep)
    for c in range(NUM_CLASSES):
        cells = [CITYSCAPES_CLASSES[c]]
        for cid in CONDITION_NAMES:
            v = iou_per_cond[cid][c].item()
            cells.append('—' if math.isnan(v) else f'{v*100:.1f}')
        v_all = iou_overall[c].item()
        cells.append('—' if math.isnan(v_all) else f'{v_all*100:.1f}')
        lines.append('| ' + ' | '.join(cells) + ' |')

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'✅ Markdown written: {md_path}')
    print(f'   Overall mIoU: {miou_overall*100:.2f}%')
    for cid in CONDITION_NAMES:
        print(f'   {CONDITION_NAMES[cid]:6s}: {miou_per_cond[cid]*100:.2f}%')


if __name__ == '__main__':
    main()
