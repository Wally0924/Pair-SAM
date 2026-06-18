#!/usr/bin/env python
"""定性比較圖（max-gain 選圖版）：聚焦 reference 的效果 = A2(no-ref) vs FULL。

作法（準則 B：每條件挑 FULL−A2 落差最大的一張）：
  1. 每個惡劣條件取前 N(=25) 張 val 影像。
  2. 每張用 A2 與 FULL 各推論一次（config-aware，與 eval_e1 同源）。
  3. 算每張 per-image mIoU（對該張 GT 出現的類別取 IoU 均值，排除 ignore/invalid）。
  4. 選圖分數 = FULL_mIoU − A2_mIoU，挑該條件最大的一張。
  5. 4 條件組成定性圖：欄位 = Input / GT / A2(no-ref) / FULL（不含 R1，聚焦 reference）。

誠實性護欄：
  - 全候選分數存 CSV（qualitative_selection_scores.csv），選圖可稽核。
  - 圖標題明確標示為「largest reference gain」範例（非典型表現）。
  - 夜間落差本就小（整體 +1.7），子標題如實顯示，不誇大。

VRAM 安全：一次只載入一個模型（A2 全跑完再載 FULL）。
GPU 需求：最多 2×N×4 次推論；請在 Batch 2 訓練結束、GPU 空閒後執行。
"""
import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.eval._eval_common import (
    load_weather_sam_from_ablation, make_batched_input,
    colorize_19class, denorm_image, CONDITION_NAMES, DEFAULT_VAL_CSV,
)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits
from utils.weather_dataloader import WeatherSegmentationDataset

OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation')
SAVE_FIG = os.path.join(OUT_ROOT, 'fig_ablation_qualitative_maxgain.png')
SAVE_CSV = os.path.join(OUT_ROOT, 'qualitative_selection_scores.csv')
DEVICE = 'cuda'
NUM_CLASSES = 19
IGNORE = 255
TOPN = 25  # 每條件取前 N 張

# 只比較 A2 vs FULL（聚焦 reference）。標籤 → run 目錄。
A2_DIR = 'A2_seed42'
FULL_DIR = 'R7_seed42'


@torch.no_grad()
def infer_pred(model, item):
    """單張推論 → (H,W) argmax 預測（uint8），流程同 eval_e1。"""
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
        low_res, class_ids_out, fusion_head=model.context_fusion_head,
        num_classes=NUM_CLASSES, use_lrh=getattr(model, 'use_lrh', True),
    )
    fused_hr = F.interpolate(fused, size=(1024, 1024), mode='bilinear', align_corners=False)
    return fused_hr.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def per_image_miou(pred, gt, invalid):
    """per-image mIoU：對 GT 中出現的類別取 IoU 均值，排除 ignore/invalid。"""
    gt = gt.copy()
    gt[invalid] = IGNORE
    valid = gt != IGNORE
    if not valid.any():
        return float('nan')
    ious = []
    for c in np.unique(gt[valid]):
        c = int(c)
        if c == IGNORE:
            continue
        p = (pred == c) & valid
        g = (gt == c) & valid
        union = (p | g).sum()
        if union > 0:
            ious.append((p & g).sum() / union)
    return float(np.mean(ious) * 100) if ious else float('nan')


def first_n_indices(csv_path, cid, n):
    df = pd.read_csv(csv_path)
    idxs = df.index[df['condition_id'] == cid].tolist()
    return idxs[:n]


def main():
    ds = WeatherSegmentationDataset(
        csv_file=DEFAULT_VAL_CSV, image_size=1024, mode='val', force_raw_images=True,
    )
    conds = list(CONDITION_NAMES.keys())
    cand = {cid: first_n_indices(DEFAULT_VAL_CSV, cid, TOPN) for cid in conds}
    for cid in conds:
        print(f'condition {CONDITION_NAMES[cid]}: {len(cand[cid])} 張候選')

    # 預載 items（input/GT 共用），與 GT/invalid numpy
    items, gts, invs = {}, {}, {}
    for cid in conds:
        for idx in cand[cid]:
            it = ds[idx]
            items[(cid, idx)] = it
            gts[(cid, idx)] = it['gt_mask'].cpu().numpy()
            invs[(cid, idx)] = it['invalid_mask'].cpu().numpy().astype(bool)

    # 逐模型推論並記錄 per-image mIoU + 暫存 pred（uint8，輕量）
    preds = {}   # (label, cid, idx) -> pred uint8
    miou = {}    # (label, cid, idx) -> float
    for label, run_dir in [('A2', A2_DIR), ('FULL', FULL_DIR)]:
        ckpt = os.path.join(OUT_ROOT, run_dir, 'weather_sam_best_latest.pth')
        cfg_path = os.path.join(OUT_ROOT, run_dir, 'ablation_config.json')
        print(f'\n[load] {label} ← {run_dir}')
        model, cfg = load_weather_sam_from_ablation(ckpt, cfg_path, device=DEVICE)
        model.eval()
        for cid in conds:
            for idx in cand[cid]:
                pred = infer_pred(model, items[(cid, idx)])
                preds[(label, cid, idx)] = pred
                miou[(label, cid, idx)] = per_image_miou(pred, gts[(cid, idx)], invs[(cid, idx)])
        del model
        torch.cuda.empty_cache()

    # 選圖（每條件 max FULL−A2）+ 寫稽核 CSV
    selected = {}
    rows = []
    for cid in conds:
        best_idx, best_gain = None, -1e9
        for idx in cand[cid]:
            a2 = miou[('A2', cid, idx)]
            fu = miou[('FULL', cid, idx)]
            gain = (fu - a2) if (a2 == a2 and fu == fu) else float('-inf')
            rows.append({
                'condition': CONDITION_NAMES[cid], 'image_index': idx,
                'A2_miou': round(a2, 2), 'FULL_miou': round(fu, 2),
                'gain': round(gain, 2) if gain != float('-inf') else '',
            })
            if gain > best_gain:
                best_gain, best_idx = gain, idx
        selected[cid] = best_idx
        print(f'[select] {CONDITION_NAMES[cid]}: idx={best_idx}  '
              f'A2={miou[("A2",cid,best_idx)]:.1f}  FULL={miou[("FULL",cid,best_idx)]:.1f}  '
              f'+{best_gain:.1f}')

    with open(SAVE_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'image_index', 'A2_miou', 'FULL_miou', 'gain', 'selected'])
        w.writeheader()
        for r in rows:
            cid = [k for k, v in CONDITION_NAMES.items() if v == r['condition']][0]
            r['selected'] = (r['image_index'] == selected[cid])
            w.writerow(r)
    print(f'✅ 稽核分數 → {SAVE_CSV}')

    # 繪圖：4 條件 × 5 欄（Input / GT / A2 / FULL / Reference）
    # 最右欄放晴天參考影像，讓讀者看出 FULL 比 A2 多取用了什麼跨視角資訊。
    col_titles = ['Input (Adverse)', 'Ground Truth', 'A2 (no-ref)', 'FULL', 'Reference (Clear)']
    ncol = len(col_titles)
    fig, axes = plt.subplots(len(conds), ncol, figsize=(3.2 * ncol, 3.3 * len(conds)))
    for row, cid in enumerate(conds):
        idx = selected[cid]
        it = items[(cid, idx)]
        gt_np = gts[(cid, idx)].copy()
        gt_np[invs[(cid, idx)]] = IGNORE
        a2_m = miou[('A2', cid, idx)]
        fu_m = miou[('FULL', cid, idx)]

        axes[row, 0].imshow(denorm_image(it['image']))
        axes[row, 1].imshow(colorize_19class(gt_np))
        axes[row, 2].imshow(colorize_19class(preds[('A2', cid, idx)]))
        axes[row, 3].imshow(colorize_19class(preds[('FULL', cid, idx)]))
        axes[row, 4].imshow(denorm_image(it['clear_image']))  # 晴天參考（raw 來源）

        axes[row, 0].set_ylabel(CONDITION_NAMES[cid].capitalize(), fontsize=13, fontweight='bold')
        axes[row, 2].set_xlabel(f'mIoU {a2_m:.1f}', fontsize=10)
        axes[row, 3].set_xlabel(f'mIoU {fu_m:.1f}  (+{fu_m - a2_m:.1f})',
                                fontsize=10, fontweight='bold', color='#2A6F3B')
        for col in range(ncol):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    fig.suptitle('With vs. Without Reference Image (A2 vs. FULL)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(SAVE_FIG, dpi=150)
    print(f'✅ 定性圖（max-gain）→ {SAVE_FIG}')


if __name__ == '__main__':
    main()
