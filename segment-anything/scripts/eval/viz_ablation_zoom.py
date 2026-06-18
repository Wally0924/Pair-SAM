#!/usr/bin/env python
"""定性放大圖（A2 vs FULL 改善區域）：與全幀版 fig_ablation_qualitative_maxgain.png 對比用。

作法：
  1. 讀 qualitative_selection_scores.csv 取已選的 4 個 frame（每條件 max FULL−A2 那張）。
  2. 對這 4 張重推論 A2 與 FULL（便宜，8 次推論）。
  3. 算像素級 diff = (FULL 對 & A2 錯 & valid)，即 reference 實際救回的像素。
  4. 滑動固定視窗自動定位 diff 最密集的區域（資料驅動，不手動挑，避免雙重精選）。
  5. 出圖：4 條件 × 4 欄 = Input(全幀+紅框) / A2(放大) / FULL(放大) / GT(放大)。

不覆蓋全幀版；另存 fig_ablation_qualitative_zoom.png。
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.eval._eval_common import (
    load_weather_sam_from_ablation, make_batched_input,
    colorize_19class, denorm_image, CONDITION_NAMES, DEFAULT_VAL_CSV,
)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits
from utils.weather_dataloader import WeatherSegmentationDataset

OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation')
CSV_IN = os.path.join(OUT_ROOT, 'qualitative_selection_scores.csv')
SAVE_FIG = os.path.join(OUT_ROOT, 'fig_ablation_qualitative_zoom.png')
DEVICE = 'cuda'
NUM_CLASSES = 19
IGNORE = 255
WIN = 384       # 放大視窗邊長（像素，1024 輸入下約佔 3/8）
STRIDE = 32     # 視窗搜尋步長
CONDITION_NAMES_INV = {v: k for k, v in CONDITION_NAMES.items()}

A2_DIR = 'A2_seed42'
FULL_DIR = 'R7_seed42'


@torch.no_grad()
def infer_pred(model, item):
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


def best_window(diff, win=WIN, stride=STRIDE):
    """滑動視窗找 diff(bool) 像素最多的方框，回傳 (y, x, win)。整合影像精確求和。"""
    H, W = diff.shape
    win = min(win, H, W)
    ii = np.pad(diff.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = (-1, 0, 0)
    ys = list(range(0, H - win + 1, stride)) or [0]
    xs = list(range(0, W - win + 1, stride)) or [0]
    for y in ys:
        for x in xs:
            s = ii[y + win, x + win] - ii[y, x + win] - ii[y + win, x] + ii[y, x]
            if s > best[0]:
                best = (s, y, x)
    return best[1], best[2], win, int(best[0])


def main():
    if not os.path.isfile(CSV_IN):
        raise SystemExit(f'找不到 {CSV_IN}；請先跑 viz_ablation_maxgain.py 產生選圖 CSV。')

    df = pd.read_csv(CSV_IN)
    sel = df[df['selected'].astype(str).str.lower() == 'true']
    # condition -> image_index
    chosen = {CONDITION_NAMES_INV[r['condition']]: int(r['image_index'])
              for _, r in sel.iterrows()}
    conds = list(CONDITION_NAMES.keys())
    print('已選 frame:', {CONDITION_NAMES[c]: chosen[c] for c in conds})

    ds = WeatherSegmentationDataset(
        csv_file=DEFAULT_VAL_CSV, image_size=1024, mode='val', force_raw_images=True,
    )
    items = {c: ds[chosen[c]] for c in conds}
    gts = {c: items[c]['gt_mask'].cpu().numpy() for c in conds}
    invs = {c: items[c]['invalid_mask'].cpu().numpy().astype(bool) for c in conds}

    # 推論（一次一個模型）
    preds = {}
    for label, run_dir in [('A2', A2_DIR), ('FULL', FULL_DIR)]:
        ckpt = os.path.join(OUT_ROOT, run_dir, 'weather_sam_best_latest.pth')
        cfg = os.path.join(OUT_ROOT, run_dir, 'ablation_config.json')
        print(f'[load] {label} ← {run_dir}')
        model, _ = load_weather_sam_from_ablation(ckpt, cfg, device=DEVICE)
        model.eval()
        for c in conds:
            preds[(label, c)] = infer_pred(model, items[c])
        del model
        torch.cuda.empty_cache()

    # 繪圖：4 條件 × 4 欄（Input 全幀+框 / A2 放大 / FULL 放大 / GT 放大）
    col_titles = ['Scene (zoom box)', 'A2 (no-ref)', 'FULL', 'Ground Truth']
    fig, axes = plt.subplots(len(conds), 4, figsize=(13, 3.3 * len(conds)))
    for row, c in enumerate(conds):
        gt = gts[c].copy(); gt[invs[c]] = IGNORE
        valid = gt != IGNORE
        a2p, fup = preds[('A2', c)], preds[('FULL', c)]
        diff = (fup == gt) & (a2p != gt) & valid     # FULL 救回、A2 漏的像素
        y, x, w, cnt = best_window(diff)

        scene = denorm_image(items[c]['image'])
        axes[row, 0].imshow(scene)
        axes[row, 0].add_patch(Rectangle((x, y), w, w, fill=False,
                                         edgecolor='red', linewidth=2.5))
        axes[row, 1].imshow(colorize_19class(a2p)[y:y + w, x:x + w])
        axes[row, 2].imshow(colorize_19class(fup)[y:y + w, x:x + w])
        axes[row, 3].imshow(colorize_19class(gt)[y:y + w, x:x + w])

        axes[row, 0].set_ylabel(CONDITION_NAMES[c].capitalize(), fontsize=13, fontweight='bold')
        for col in range(4):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    fig.suptitle('With vs. Without Reference, zoomed to largest A2->FULL improvement',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(SAVE_FIG, dpi=150)
    print(f'✅ 放大圖 → {SAVE_FIG}')


if __name__ == '__main__':
    main()
