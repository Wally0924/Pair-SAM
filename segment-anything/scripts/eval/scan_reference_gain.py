#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACDC val 全量掃描：有無參考影像的逐張 / 逐類別差異，以及逐張對齊置信度統計。

對照組（唯一差異為參考影像，Adapter 結構與參數量不變）：
  無參考  W2_semB_seed42  參考先驗置零、對齊置信度設為中性值 1
  有參考  FULL_seed42     完整模型

推論流程與 eval_e1_acdc_val_full.py 同源（assemble_semantic_logits + use_lrh）。
置信度取自 fusion_module.pre_align() 回傳的 'mask'，即論文式 (3.1) 的 m，
與訓練/推論時實際注入 Injector 的是同一張圖（RPM 只做逐尺度池化，不改變數值分布）。

輸出（figures_defense/reference_gain/）：
  per_image.csv                 逐張 mIoU、增益、置信度統計
  per_class_iou.csv             逐類別累加 IoU（ref / noref）
  per_class_iou_by_condition.csv 逐條件逐類別累加 IoU

用法：
    conda run -n sam_env python scripts/eval/scan_reference_gain.py
"""
import argparse, csv, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.eval._eval_common import (
    load_pair_sam_from_ablation, make_batched_input,
    CONDITION_NAMES, CITYSCAPES_CLASSES, DEFAULT_VAL_CSV,
)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits
from utils.pair_dataloader import PairSegmentationDataset

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ABL = os.path.join(ROOT, 'outputs_ablation_m2f')
OUT_DIR = os.path.join(ROOT, 'figures_defense', 'reference_gain')
NOREF_DIR, REF_DIR = 'W2_semB_seed42', 'FULL_seed42'
DEVICE, NUM_CLASSES, IGNORE = 'cuda', 19, 255
STATIC_IDS = set(range(0, 11))    # road..sky
DYNAMIC_IDS = set(range(11, 19))  # person..bicycle


@torch.no_grad()
def infer_pred(model, item):
    """單張推論 → (1024,1024) argmax 預測，流程同 eval_e1。"""
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

    fusion.pre_align() 的 Step 8 已把 confidence 快取於 _last_confidence_map，
    實際形狀為 (1, 1, 64, 64)（out_size 預設 64×64）、位於 CPU（pre_align 內
    已 .cpu()）。直接讀快取而非重呼叫 pre_align，可省下一次完整的 VGG +
    UAWarpC 前傳，且保證取到的就是該次推論實際注入 Injector 的那張圖（RPM
    只做逐尺度池化，不改變數值分布）。

    必須在 infer_pred 之後、且模型為有參考設定時呼叫；無參考設定的 forward
    因 _adapter_reference_free 為真而不執行 pre_align，快取會是上一張的殘值。
    """
    conf = model.fusion_module._last_confidence_map   # (1,1,64,64) on CPU
    conf = F.interpolate(conf, size=(1024, 1024), mode='bilinear', align_corners=False)
    return conf.squeeze().numpy()


def per_image_miou(pred, gt, invalid):
    """逐張 mIoU：只對該張 GT 出現過的類別取平均（與 viz 腳本一致）。"""
    gt = gt.copy(); gt[invalid] = IGNORE
    valid = gt != IGNORE
    if not valid.any():
        return float('nan')
    ious = []
    for c in np.unique(gt[valid]):
        c = int(c)
        if c == IGNORE:
            continue
        p, g = (pred == c) & valid, (gt == c) & valid
        u = (p | g).sum()
        if u > 0:
            ious.append((p & g).sum() / u)
    return float(np.mean(ious) * 100) if ious else float('nan')


def accumulate_iou(acc, pred, gt, invalid, key_prefix):
    """把單張的 intersection / union 累加進 acc（dict）。全域累加而非逐張平均。"""
    gt = gt.copy(); gt[invalid] = IGNORE
    valid = gt != IGNORE
    for c in range(NUM_CLASSES):
        p, g = (pred == c) & valid, (gt == c) & valid
        inter, union = int((p & g).sum()), int((p | g).sum())
        k = key_prefix + (c,)
        if k not in acc:
            acc[k] = [0, 0]
        acc[k][0] += inter
        acc[k][1] += union


def conf_stats(conf, gt, invalid):
    """回傳 (conf_mean, valid_ratio, conf_static, conf_dynamic)，僅計 GT 有效像素。"""
    gt = gt.copy(); gt[invalid] = IGNORE
    valid = gt != IGNORE
    if not valid.any():
        return (float('nan'),) * 4
    cv = conf[valid]
    st = np.isin(gt, list(STATIC_IDS)) & valid
    dy = np.isin(gt, list(DYNAMIC_IDS)) & valid
    return (
        float(cv.mean()),
        float((cv > 0).mean()),
        float(conf[st].mean()) if st.any() else float('nan'),
        float(conf[dy].mean()) if dy.any() else float('nan'),
    )


def pixel_counts(gt, invalid):
    """回傳 (n_valid, n_static, n_dynamic) 像素數，僅依 GT / invalid_mask，與模型無關。
    供還原全域像素加權平均使用（區別於 per_image.csv 其餘欄位的逐張等權平均）。
    """
    gt = gt.copy(); gt[invalid] = IGNORE
    valid = gt != IGNORE
    st = np.isin(gt, list(STATIC_IDS)) & valid
    dy = np.isin(gt, list(DYNAMIC_IDS)) & valid
    return int(valid.sum()), int(st.sum()), int(dy.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='只掃前 N 張（0 = 全部 406 張）；供冒煙測試用')
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = PairSegmentationDataset(csv_file=DEFAULT_VAL_CSV, image_size=1024,
                                 mode='val', force_raw_images=True)
    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f'掃描 {n} 張（全集 {len(ds)} 張）')

    # GT 與置信度與模型無關，先備妥；置信度需模型內的凍結對齊網路，故於首個模型載入後計算
    gts, invs, conds, items = {}, {}, {}, {}
    for i in range(n):
        it = ds[i]
        items[i] = it
        gts[i] = it['gt_mask'].cpu().numpy()
        invs[i] = it['invalid_mask'].cpu().numpy().astype(bool)
        conds[i] = CONDITION_NAMES[int(it['condition_id'])]

    mious, acc, cstats = {}, {}, {}
    for label, run in [('ref', REF_DIR), ('noref', NOREF_DIR)]:
        ckpt = os.path.join(ABL, run, 'weather_sam_best_latest.pth')
        print(f'\n── 載入 {label} ({run})…', flush=True)
        model, _ = load_pair_sam_from_ablation(ckpt, device=DEVICE)
        for i in range(n):
            p = infer_pred(model, items[i])
            mious[(label, i)] = per_image_miou(p, gts[i], invs[i])
            accumulate_iou(acc, p, gts[i], invs[i], (label, 'all'))
            accumulate_iou(acc, p, gts[i], invs[i], (label, conds[i]))
            # 置信度只需算一次，且必須用有參考的模型（noref 模型不跑 pre_align）
            if label == 'ref':
                cstats[i] = conf_stats(read_confidence(model), gts[i], invs[i])
            if (i + 1) % 50 == 0:
                print(f'   {i + 1}/{n}', flush=True)
        del model
        torch.cuda.empty_cache()

    # per_image.csv
    with open(os.path.join(args.out, 'per_image.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_index', 'condition', 'noref_miou', 'ref_miou', 'gain',
                    'conf_mean', 'conf_valid_ratio', 'conf_static', 'conf_dynamic',
                    'n_valid', 'n_static', 'n_dynamic'])
        for i in range(n):
            a, b = mious[('ref', i)], mious[('noref', i)]
            cm, cv, cs, cd = cstats[i]
            nv, ns, nd = pixel_counts(gts[i], invs[i])
            w.writerow([i, conds[i], round(b, 3), round(a, 3), round(a - b, 3),
                        round(cm, 4), round(cv, 4), round(cs, 4), round(cd, 4),
                        nv, ns, nd])

    # per_class_iou.csv 與 per_class_iou_by_condition.csv
    def dump(path, scope_filter, extra_cols):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(extra_cols + ['setting', 'class_id', 'class_name',
                                     'intersection', 'union', 'iou'])
            for (label, scope, c), (inter, union) in sorted(acc.items()):
                if not scope_filter(scope):
                    continue
                iou = round(inter / union * 100, 3) if union > 0 else ''
                pre = [] if not extra_cols else [scope]
                w.writerow(pre + [label, c, CITYSCAPES_CLASSES[c], inter, union, iou])

    dump(os.path.join(args.out, 'per_class_iou.csv'), lambda s: s == 'all', [])
    dump(os.path.join(args.out, 'per_class_iou_by_condition.csv'),
         lambda s: s != 'all', ['condition'])

    # 交叉核對用摘要
    for label in ('ref', 'noref'):
        ious = [i / u * 100 for (l, s, _), (i, u) in acc.items()
                if l == label and s == 'all' and u > 0]
        print(f'{label:>6} 全集 mIoU = {np.mean(ious):.2f}%')
    print(f'\n輸出目錄：{args.out}')


if __name__ == '__main__':
    main()
