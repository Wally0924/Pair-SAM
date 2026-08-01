#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q6 定性圖：有無參考影像的並列比較，逐張單獨輸出（供人工挑選後放入論文）。

對照組（唯一差異為參考影像，Adapter 結構與參數量不變）：
  無參考  W2_semB_seed42  參考先驗置零、對齊置信度設為中性值 1
  有參考  FULL_seed42     完整模型

樣本選取直接讀 Task 1 全量掃描的產出 `figures_defense/reference_gain/per_image.csv`
（欄位：image_index, condition, noref_miou, ref_miou, gain, conf_mean, ...），
依 `gain` 排序取每條件 top-K（正面案例）與 bottom-K（負面案例），
不再自行推論排序，也不重跑全量 mIoU 計算——只對被選中的約 40 張樣本重新推論。

推論流程與 eval_e1_acdc_val_full.py 同源（assemble_semantic_logits + use_lrh），
故定性預測與論文表格的 mIoU 一致。

輸出（figures_defense/q6_singles/）：
  <cond>_pos_rank<k>_idx<i>_gain<g>_1input.png     惡劣天候輸入
  <cond>_pos_rank<k>_idx<i>_gain<g>_2reference.png 參考影像（晴天）
  <cond>_pos_rank<k>_idx<i>_gain<g>_3gt.png        真值標註
  <cond>_pos_rank<k>_idx<i>_gain<g>_4noref.png     無參考預測
  <cond>_pos_rank<k>_idx<i>_gain<g>_5ref.png       有參考預測
  <cond>_pos_rank<k>_idx<i>_gain<g>_6diff.png      差異圖（綠=參考修正、紅=參考破壞）
  <cond>_pos_rank<k>_idx<i>_gain<g>_7conf.png      對齊置信度色階圖（0 藍 → 1 黃）
  （負面案例檔名前綴為 <cond>_neg_rank<k>_...）
  selection_scores.csv                              入選樣本分數，供稽核

誠實性說明：
  本腳本依 Task 1 全量掃描的 per-image mIoU 增益排序選圖，正面案例取增益最大的
  前 K 張、負面案例取增益最小（最負）的前 K 張。選圖準則會寫進檔名與 CSV，
  論文採用時須於圖說載明「此為參考影像增益最大／最小之樣本」。
  第 4.5 節的量化結論為：參考補償具條件依賴性，於雨天與雪天為正、霧天與夜間為負；
  負面案例集中在霧天與夜間屬預期結果，不予迴避。

用法：
    conda run -n sam_env python scripts/eval/viz_reference_gain_singles.py \
        --conditions fog rain snow night --topk 5 --bottomk 5
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.eval._eval_common import (  # noqa: E402
    load_pair_sam_from_ablation, make_batched_input,
    colorize_19class, denorm_image, CONDITION_NAMES, DEFAULT_VAL_CSV,
)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits  # noqa: E402
from utils.pair_dataloader import PairSegmentationDataset  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ABL = os.path.join(ROOT, 'outputs_ablation_m2f')
OUT_DIR = os.path.join(ROOT, 'figures_defense', 'q6_singles')
SCAN_CSV = os.path.join(ROOT, 'figures_defense', 'reference_gain', 'per_image.csv')

NOREF_DIR = 'W2_semB_seed42'   # ref=False, cond=True
REF_DIR = 'FULL_seed42'        # ref=True,  cond=True

DEVICE = 'cuda'
NUM_CLASSES = 19
IGNORE = 255
NAME2CID = {v: k for k, v in CONDITION_NAMES.items()}


@torch.no_grad()
def infer_pred(model, item):
    """單張推論 → (H,W) argmax 預測，流程同 eval_e1。"""
    batch = {
        'image': item['image'].unsqueeze(0),
        'clear_image': item['clear_image'].unsqueeze(0),
        'gt_mask': item['gt_mask'].unsqueeze(0),
        'invalid_mask': item['invalid_mask'].unsqueeze(0),
        'text_prompts': [item['text_prompts']],
        'original_size': [item['original_size']],
        'condition_id': item['condition_id'].unsqueeze(0),
    }
    out = model(make_batched_input(batch, DEVICE))
    low_res = out[0]['low_res_logits'].squeeze(0)
    fused = assemble_semantic_logits(
        low_res, out[0]['class_ids'], fusion_head=model.context_fusion_head,
        num_classes=NUM_CLASSES, use_lrh=getattr(model, 'use_lrh', True),
    )
    hr = F.interpolate(fused, size=(1024, 1024), mode='bilinear', align_corners=False)
    return hr.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def read_confidence(model):
    """讀取上一次 forward 所產生的對齊置信度圖 m，上採至 (1024,1024)。

    必須在 infer_pred 之後、且模型為有參考設定（FULL_seed42）時呼叫；
    無參考設定的 forward 因 _adapter_reference_free 為真而不執行 pre_align，
    快取會是上一張的殘值，故本腳本只在 REF_DIR 模型上讀取。
    """
    conf = model.fusion_module._last_confidence_map   # (1,1,64,64) on CPU
    conf = F.interpolate(conf, size=(1024, 1024), mode='bilinear', align_corners=False)
    return conf.squeeze().numpy()


def save_img(arr, path, cmap=None):
    """無邊框、原始解析度輸出單張圖。"""
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(arr, cmap=cmap, interpolation='nearest')
    fig.savefig(path, dpi=100, pad_inches=0)
    plt.close(fig)


def save_conf(conf, path):
    """置信度色階圖：0（藍）→ 1（黃），與 ch4_confidence_maps.png 的呈現一致。"""
    h, w = conf.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.imshow(conf, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    fig.savefig(path, dpi=100, pad_inches=0)
    plt.close(fig)


def diff_map(pred_ref, pred_noref, gt, invalid):
    """綠 = 參考修正（有參考對、無參考錯）；紅 = 參考破壞；灰 = 兩者一致。"""
    gt = gt.copy()
    gt[invalid] = IGNORE
    valid = gt != IGNORE
    ok_ref, ok_no = (pred_ref == gt) & valid, (pred_noref == gt) & valid
    out = np.full((*gt.shape, 3), 235, np.uint8)
    out[~valid] = 255
    out[ok_ref & ~ok_no] = [40, 170, 70]    # 參考救回
    out[~ok_ref & ok_no] = [200, 50, 50]    # 參考弄壞
    return out


def select_samples(scan_df, conditions, topk, bottomk):
    """依 per_image.csv 的 gain 排序，回傳 {(cond, tag, rank): row} 的入選清單。

    tag='pos' 取 gain 最大的前 topk 張；tag='neg' 取 gain 最小（最負）的前
    bottomk 張。若某條件全部樣本 gain 皆為正，neg 榜仍照排序輸出前 bottomk
    張（即該條件「最不正」的樣本），並如實反映於報告，不特別過濾。
    """
    selected = []
    for cond in conditions:
        sub = scan_df[scan_df['condition'] == cond].sort_values('gain', ascending=False)
        pos = sub.head(topk)
        neg = sub.tail(bottomk).sort_values('gain', ascending=True)
        for rank, (_, row) in enumerate(pos.iterrows(), 1):
            selected.append((cond, 'pos', rank, row))
        for rank, (_, row) in enumerate(neg.iterrows(), 1):
            selected.append((cond, 'neg', rank, row))
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+', default=['fog', 'rain', 'snow', 'night'],
                    help='要輸出的條件；預設四種全開')
    ap.add_argument('--topk', type=int, default=5, help='每條件輸出增益最大的前 K 張（正面案例）')
    ap.add_argument('--bottomk', type=int, default=5, help='每條件輸出增益最小的前 K 張（負面案例）')
    ap.add_argument('--scan-csv', default=SCAN_CSV, help='Task 1 全量掃描輸出的 per_image.csv')
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = PairSegmentationDataset(csv_file=DEFAULT_VAL_CSV, image_size=1024,
                                 mode='val', force_raw_images=True)
    scan_df = pd.read_csv(args.scan_csv)

    selected = select_samples(scan_df, args.conditions, args.topk, args.bottomk)
    indices = sorted({int(row['image_index']) for _, _, _, row in selected})
    print(f'選中 {len(selected)} 組樣本（{len(indices)} 張不重複影像）')

    items, gts, invs = {}, {}, {}
    for i in indices:
        it = ds[i]
        items[i] = it
        gts[i] = it['gt_mask'].cpu().numpy()
        invs[i] = it['invalid_mask'].cpu().numpy().astype(bool)

    # 一次載入一個模型，跑完全部選中樣本再釋放（VRAM 安全）；置信度只能由
    # 有參考模型（REF_DIR）讀取，noref 模型的 forward 不執行 pre_align。
    preds, confs = {}, {}
    for label, run in [('noref', NOREF_DIR), ('ref', REF_DIR)]:
        ckpt = os.path.join(ABL, run, 'weather_sam_best_latest.pth')
        print(f'\n── 載入 {label} ({run})…', flush=True)
        model, _ = load_pair_sam_from_ablation(ckpt, device=DEVICE)
        for i in indices:
            preds[(label, i)] = infer_pred(model, items[i])
            if label == 'ref':
                confs[i] = read_confidence(model)
        del model
        torch.cuda.empty_cache()

    rows = []
    print(f'\n{"=" * 66}\n各條件依 Task 1 掃描結果選圖（正面 top-{args.topk} / 負面 bottom-{args.bottomk}）\n{"=" * 66}')
    for cond in args.conditions:
        for tag in ('pos', 'neg'):
            group = [(c, t, r, row) for (c, t, r, row) in selected if c == cond and t == tag]
            if not group:
                continue
            print(f'\n### {cond} [{tag}]')
            for _, _, rank, row in group:
                i = int(row['image_index'])
                g = float(row['gain'])
                noref_m, ref_m = float(row['noref_miou']), float(row['ref_miou'])
                conf_mean = row['conf_mean']
                print(f'  rank{rank}  idx={i:<5} 無參考 {noref_m:5.1f} → '
                      f'有參考 {ref_m:5.1f}   增益 {g:+.2f}   conf_mean={conf_mean}')
                rows.append(dict(condition=cond, tag=tag, rank=rank, image_index=i,
                                 noref_miou=noref_m, ref_miou=ref_m, gain=g,
                                 conf_mean=conf_mean))

                it = items[i]
                base = os.path.join(args.out, f'{cond}_{tag}_rank{rank}_idx{i}_gain{g:+.2f}')
                gt, inv = gts[i], invs[i]
                gt_vis = gt.copy()
                gt_vis[inv] = IGNORE

                save_img(denorm_image(it['image']), f'{base}_1input.png')
                save_img(denorm_image(it['clear_image']), f'{base}_2reference.png')
                save_img(colorize_19class(gt_vis), f'{base}_3gt.png')
                save_img(colorize_19class(preds[('noref', i)]), f'{base}_4noref.png')
                save_img(colorize_19class(preds[('ref', i)]), f'{base}_5ref.png')
                save_img(diff_map(preds[('ref', i)], preds[('noref', i)], gt, inv),
                         f'{base}_6diff.png')
                save_conf(confs[i], f'{base}_7conf.png')

    csv_path = os.path.join(args.out, 'selection_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['condition'], r['tag'], r['rank'])))

    print(f'\n輸出目錄：{args.out}')
    print(f'入選樣本分數：{csv_path}')
    print('\n提醒：pos 為「參考影像增益最大」樣本（best case），'
          '\n      neg 為「參考影像增益最小（含負值）」樣本（worst case，供反面案例佐證）。'
          '\n      論文採用時圖說須載明選圖準則，並與第 4.5 節的條件依賴結論一致。')


if __name__ == '__main__':
    main()
