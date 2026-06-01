"""
E1-paper: ACDC val 論文口徑評估（Refign / CMA Table 1, 4 對齊版）
============================================================
與 eval_e1_acdc_val_full.py 的差異：
  * 評估解析度從 1024x1024 改為 ACDC 原始尺寸（1080x1920）。
  * GT 與 invalid_mask 直接從 CSV 的 gt_path / invalid_mask 欄位讀取原始 PNG，
    不經過 dataloader 的 1024x1024 resize（避免 GT 邊界資訊損失）。
  * fused_logits (1, 19, 256, 256) 直接 bilinear 上採到 (H_orig, W_orig)。
    dataloader 用單純 cv2.resize 把 1080x1920 影像壓成 1024x1024（無 aspect-ratio
    padding），所以反向直接 interpolate 不需要 postprocess_masks 去 padding。
  * 輸出檔案：e1_acdc_val_paper_results.{json,md}

用途：拿到能直接與 CMA Table 5（SegFormer val 67.2%）或 Refign Table 4（DAFormer
val 65.0%）並列的數字。
"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (  # noqa: E402
    load_weather_sam_model, build_acdc_val_loader, make_batched_input,
    CONDITION_NAMES, CITYSCAPES_CLASSES, OUTPUT_ROOT, DEFAULT_CKPT,
)

NUM_CLASSES = 19
IGNORE_INDEX = 255
DEVICE = 'cuda'


def iou_from_confusion(cm: torch.Tensor) -> torch.Tensor:
    tp    = cm.diag().float()
    fp    = cm.sum(dim=0).float() - tp
    fn    = cm.sum(dim=1).float() - tp
    denom = tp + fp + fn
    return torch.where(denom > 0, tp / denom, torch.full_like(tp, float('nan')))


def load_native_gt(gt_path: str, invalid_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    """從磁碟讀原始解析度 GT (H, W) int64 + invalid mask (H, W) bool。"""
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f'GT not found: {gt_path}')
    gt = gt.astype(np.int64)
    if invalid_path and Path(invalid_path).is_file():
        inv = cv2.imread(invalid_path, cv2.IMREAD_GRAYSCALE)
        inv = (inv != 0)
    else:
        inv = np.zeros_like(gt, dtype=bool)
    return gt, inv


def main():
    model = load_weather_sam_model(DEFAULT_CKPT, device=DEVICE)
    loader = build_acdc_val_loader()
    # 透過 dataset.data 取得原始 CSV row（順序與 loader 一致，shuffle=False）
    csv_df = loader.dataset.data.reset_index(drop=True)

    cm_overall  = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    cm_per_cond = {cid: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
                   for cid in CONDITION_NAMES}
    sample_counts = {cid: 0 for cid in CONDITION_NAMES}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc='E1-paper ACDC val (native res)')):
            # ── 1. forward + scatter + context_fusion_head ──
            batched_input = make_batched_input(batch, DEVICE)
            outputs = model(batched_input)
            low_res = outputs[0]['low_res_logits'].squeeze(0)            # (K, 256, 256)
            class_ids_out = outputs[0]['class_ids']
            full = torch.full(
                (1, NUM_CLASSES, 256, 256), -10.0,
                device=DEVICE, dtype=low_res.dtype,
            )
            for k, c in enumerate(class_ids_out):
                full[0, c] = low_res[k]
            fused = model.context_fusion_head(full)                       # (1, 19, 256, 256)

            # ── 2. 讀原始解析度 GT 與 invalid_mask ──
            row = csv_df.iloc[idx]
            gt_np, inv_np = load_native_gt(
                str(row['gt_path']),
                str(row['invalid_mask']) if 'invalid_mask' in row else None,
            )
            H, W = gt_np.shape  # ACDC: 1080 x 1920

            # ── 3. 上採至 GT 原始解析度 ──
            fused_hr = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
            pred = fused_hr.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)  # (H, W)

            # ── 4. 過濾 ignore（GT==255 或 invalid）──
            gt_used = gt_np.copy()
            gt_used[inv_np] = IGNORE_INDEX
            valid = gt_used != IGNORE_INDEX
            if not valid.any():
                continue
            g = gt_used[valid]
            p = pred[valid]

            # ── 5. 累積混淆矩陣 ──
            cm_step = torch.bincount(
                torch.from_numpy(g * NUM_CLASSES + p),
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            cm_overall = cm_overall + cm_step
            cid = int(batch['condition_id'][0].item())
            cm_per_cond[cid] = cm_per_cond[cid] + cm_step
            sample_counts[cid] += 1

    # ── IoU 計算 ──
    iou_overall = iou_from_confusion(cm_overall)
    iou_per_cond = {cid: iou_from_confusion(cm) for cid, cm in cm_per_cond.items()}

    def nanmean(t):
        return float(torch.nanmean(t).item()) if not torch.isnan(t).all() else float('nan')

    miou_overall = nanmean(iou_overall)
    miou_per_cond = {cid: nanmean(iou_per_cond[cid]) for cid in CONDITION_NAMES}

    # ── JSON 輸出 ──
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / 'e1_acdc_val_paper_results.json'
    json_data = {
        'protocol': 'paper (native 1080x1920, GT not downsampled)',
        'checkpoint': str(Path(DEFAULT_CKPT).name),
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
    print(f'✅ JSON: {json_path}')

    # ── Markdown 輸出 ──
    md_path = OUTPUT_ROOT / 'e1_acdc_val_paper_results.md'
    lines = []
    lines.append('# E1-paper: WeatherSAM v15 (E27) — ACDC val (Paper Protocol)')
    lines.append('')
    lines.append(f'**Checkpoint:** `{Path(DEFAULT_CKPT).name}`')
    lines.append(f'**Date:** {datetime.now().strftime("%Y-%m-%d")}')
    lines.append('**Protocol:** native 1080×1920, GT not downsampled '
                 '(Refign / CMA ablation-table 對齊)')
    lines.append(f'**Samples:** {sum(sample_counts.values())} '
                 + '(' + ', '.join(f'{CONDITION_NAMES[cid]}={n}'
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
    lines.append('| Class | Fog | Rain | Snow | Night | All |')
    lines.append('|-------|----:|-----:|-----:|------:|----:|')
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
    print(f'✅ Markdown: {md_path}')
    print(f'   Overall mIoU: {miou_overall*100:.2f}%')
    for cid in CONDITION_NAMES:
        print(f'   {CONDITION_NAMES[cid]:6s}: {miou_per_cond[cid]*100:.2f}%')


if __name__ == '__main__':
    main()
