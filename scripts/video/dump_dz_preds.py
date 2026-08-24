"""用論文 checkpoint 重跑 Dark Zurich val 50 張，匯出逐張 trainId 預測與逐張混淆矩陣。

評估流程完全沿用 scripts/eval/eval_e1_acdc_val_full.py（1024x1024 logits、
invalid_mask 過濾），確保與論文 Table「Generalization」的 54.2 同一口徑。

用法：
    python scripts/video/dump_dz_preds.py \
        --ckpt segment-anything/outputs_ablation_m2f/FULL_seed42/weather_sam_best_latest.pth
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SA = Path(__file__).resolve().parents[2] / 'segment-anything'
sys.path.insert(0, str(SA))
sys.path.insert(0, str(SA / 'scripts' / 'eval'))
from _eval_common import (load_pair_sam_from_ablation, build_acdc_val_loader,  # noqa: E402
                          make_batched_input)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits  # noqa: E402

NUM_CLASSES = 19
IGNORE = 255
DEVICE = 'cuda'
CSV = str(Path(__file__).resolve().parents[2] / 'Datasets' / 'darkzurich_adverse_ref_rgb_val.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default=CSV)
    ap.add_argument('--out', default=os.path.expanduser('~/Downloads/figures/_ours_pred_dz_paper'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = [os.path.basename(p) for p in pd.read_csv(args.csv)['image_path']]

    model, cfg = load_pair_sam_from_ablation(args.ckpt, None, device=DEVICE)
    print(f'[dump] config: {cfg}')
    loader = build_acdc_val_loader(args.csv)

    cms = []                                   # 逐張混淆矩陣，供影片折線做 running mIoU
    cm_total = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='DZ val')):
            outputs = model(make_batched_input(batch, DEVICE))
            low_res = outputs[0]['low_res_logits'].squeeze(0)
            fused = assemble_semantic_logits(
                low_res, outputs[0]['class_ids'],
                fusion_head=model.context_fusion_head,
                num_classes=NUM_CLASSES,
                use_lrh=getattr(model, 'use_lrh', True),
            )
            pred = F.interpolate(fused, size=(1024, 1024), mode='bilinear',
                                 align_corners=False).argmax(1).squeeze(0)

            gt = batch['gt_mask'][0].to(DEVICE).long().clone()
            gt[batch['invalid_mask'][0].to(DEVICE)] = IGNORE
            m = gt != IGNORE
            cm = torch.bincount((gt[m].cpu().long() * NUM_CLASSES + pred[m].cpu().long()),
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            cms.append(cm.numpy())
            cm_total += cm

            # 存回原始 1920x1080 供影片使用（模型輸入為非等比 resize，故直接還原）
            p = pred.cpu().numpy().astype(np.uint8)
            Image.fromarray(p).resize((1920, 1080), Image.NEAREST).save(
                os.path.join(args.out, names[i]))

    np.save(os.path.join(args.out, '_per_frame_cm.npy'), np.stack(cms))

    inter = np.diag(cm_total.numpy()).astype(float)
    union = cm_total.numpy().sum(1) + cm_total.numpy().sum(0) - inter
    present = cm_total.numpy().sum(1) > 0
    print(f'aggregate mIoU over {present.sum()} present classes = '
          f'{np.mean(inter[present] / union[present]) * 100:.2f}   (paper: 54.2)')


if __name__ == '__main__':
    main()
