# v15 權重評估實驗實作計畫（E1 / E4 / E5）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 `best_E18_mIoU65.06_LR4.6e-05.pth` 權重，以 inference-only 方式產出 3 項 NCAR2026 論文素材：ACDC val 完整評估表（E1）、定性比較圖（E4）、UAWarpC warp 與 confidence 可視化（E5）。

**Architecture:** 4 個新檔案於 `segment-anything/scripts/eval/`，全部 read-only inference，不動 trainer/model 程式碼。共用 `_eval_common.py` 提供 model loader、ACDC val dataloader、Cityscapes 配色。

**Tech Stack:** PyTorch 2.x，matplotlib 3.x，conda env `sam_env`

---

## File Map

| 動作 | 檔案 | 責任 |
|------|------|------|
| 新增 | `segment-anything/scripts/eval/__init__.py` | 空檔，讓目錄成為 package |
| 新增 | `segment-anything/scripts/eval/_eval_common.py` | model loader、val loader、調色盤、輸出路徑 |
| 新增 | `segment-anything/scripts/eval/eval_e1_acdc_val_full.py` | E1 評估腳本 |
| 新增 | `segment-anything/scripts/eval/viz_e4_qualitative.py` | E4 視覺化腳本 |
| 新增 | `segment-anything/scripts/eval/viz_e5_warp_confidence.py` | E5 視覺化腳本 |
| 新增資料夾 | `docs/experiments/v15-eval-2026-05-14/` | 實驗產出位置 |

---

## Task 1：共用工具 `_eval_common.py`

**Files:**
- Create: `segment-anything/scripts/eval/__init__.py`
- Create: `segment-anything/scripts/eval/_eval_common.py`

- [ ] **Step 1：建立 package 與輸出資料夾**

```bash
mkdir -p segment-anything/scripts/eval
touch segment-anything/scripts/eval/__init__.py
mkdir -p docs/experiments/v15-eval-2026-05-14
```

- [ ] **Step 2：寫 `_eval_common.py`（共用 model loader / val loader / 調色盤）**

```python
# segment-anything/scripts/eval/_eval_common.py
"""共用工具：v15 checkpoint 載入、ACDC val dataloader、Cityscapes 19-class 調色盤。"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# 讓 script 可從 segment-anything 根目錄被執行（含 utils 與 segment_anything 兩個 import path）
_THIS = Path(__file__).resolve()
_SEGANY_ROOT = _THIS.parents[2]   # .../segment-anything
if str(_SEGANY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGANY_ROOT))

from segment_anything.build_weather_sam import build_weather_sam_vit_h
from utils.weather_dataloader import WeatherSegmentationDataset


# ── 預設路徑 ───────────────────────────────────────────────
DEFAULT_CKPT = str(
    _SEGANY_ROOT / "outputs_weather_sam_mask2former_testv15" /
    "best_E18_mIoU65.06_LR4.6e-05.pth"
)
DEFAULT_VAL_CSV = str(
    _SEGANY_ROOT.parent / "Datasets" / "acdc_adverse_ref_rgb_val.csv"
)
OUTPUT_ROOT = _SEGANY_ROOT.parent / "docs" / "experiments" / "v15-eval-2026-05-14"

# ── ACDC condition 對照 ────────────────────────────────────
CONDITION_NAMES = {0: 'fog', 1: 'rain', 2: 'snow', 3: 'night'}

# ── Cityscapes 19-class 標準調色盤（與 ACDC GT trainIds 對齊）──
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],  # road
    [244,  35, 232],  # sidewalk
    [ 70,  70,  70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170,  30],  # traffic light
    [220, 220,   0],  # traffic sign
    [107, 142,  35],  # vegetation
    [152, 251, 152],  # terrain
    [ 70, 130, 180],  # sky
    [220,  20,  60],  # person
    [255,   0,   0],  # rider
    [  0,   0, 142],  # car
    [  0,   0,  70],  # truck
    [  0,  60, 100],  # bus
    [  0,  80, 100],  # train
    [  0,   0, 230],  # motorcycle
    [119,  11,  32],  # bicycle
], dtype=np.uint8)


def colorize_19class(mask: np.ndarray) -> np.ndarray:
    """19-class trainID mask (H, W) → RGB (H, W, 3)。255 視為 ignore，輸出黑色。"""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id in range(19):
        color[mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
    return color


def load_v15_model(ckpt_path: str = DEFAULT_CKPT, device: str = 'cuda'):
    """載入 best_E18 v15 checkpoint，啟用 pre-hook cross-attn adapter。

    strict=False：v5 後續加了 `_last_kv_keep_ratio` 等 buffer，舊 ckpt 不含此欄。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_weather_sam_vit_h(num_classes=19, checkpoint=None)
    state = torch.load(ckpt_path, map_location='cpu')
    # 容忍多種儲存格式
    if isinstance(state, dict) and 'model_state_dict' in state:
        sd = state['model_state_dict']
    elif isinstance(state, dict) and 'model' in state:
        sd = state['model']
    else:
        sd = state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[load_v15_model] 忽略 {len(unexpected)} 個多餘鍵（OK）")
    if missing:
        print(f"[load_v15_model] 缺少 {len(missing)} 個鍵（多為新增 buffer，OK）")
    model.enable_vgg_adapter('pre')
    return model.to(device).eval()


def build_acdc_val_loader(csv_path: str = DEFAULT_VAL_CSV,
                           batch_size: int = 1, num_workers: int = 2):
    """ACDC val DataLoader，batch_size=1 簡化 per-condition 統計。"""
    ds = WeatherSegmentationDataset(
        csv_file=csv_path, image_size=1024, mode='val', force_raw_images=True,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=WeatherSegmentationDataset.collate_fn,
    )


def make_batched_input(batch: dict, device: str) -> list:
    """把 dataloader 出來的 batch dict 轉成 WeatherSAM.forward 需要的 list[dict]。
    與 weather_trainer.py 中的轉換邏輯保持一致。
    """
    bs = batch['gt_mask'].shape[0]
    out = []
    for i in range(bs):
        item = {
            'image':        batch['image'][i].to(device),
            'clear_image':  batch['clear_image'][i].to(device),
            'text_prompts': batch['text_prompts'][i],
            'original_size': batch['original_size'][i],
            'condition_id': batch['condition_id'][i],
        }
        out.append(item)
    return out


def pick_first_per_condition(csv_path: str = DEFAULT_VAL_CSV) -> dict:
    """從 ACDC val csv 為每個 condition_id 找出首個 row 的索引（CSV 內順序）。
    回傳：{0: idx_fog, 1: idx_rain, 2: idx_snow, 3: idx_night}
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    picked = {}
    for cid in CONDITION_NAMES:
        rows = df.index[df['condition_id'] == cid].tolist()
        if not rows:
            raise ValueError(f'No sample for condition_id={cid}')
        picked[cid] = rows[0]
    return picked


def denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    """把 dataloader 輸出（0~255 範圍 float tensor）轉成 matplotlib 友善的 uint8 RGB。"""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
```

- [ ] **Step 3：smoke test — 確認 import 與載入成功**

執行：
```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -c "
import sys; sys.path.insert(0, 'segment-anything')
from scripts.eval._eval_common import (
    load_v15_model, build_acdc_val_loader,
    pick_first_per_condition, denorm_image,
    CONDITION_NAMES, CITYSCAPES_PALETTE, colorize_19class,
)
import numpy as np
print('palette shape:', CITYSCAPES_PALETTE.shape)
m = load_v15_model(device='cuda')
print('model loaded:', type(m).__name__)
ld = build_acdc_val_loader()
batch = next(iter(ld))
print('batch keys:', sorted(batch.keys()))
print('condition_id:', batch['condition_id'].item())
print('colorize ok:', colorize_19class(np.zeros((8,8), dtype=np.int64)).shape)
picked = pick_first_per_condition()
print('picked indices:', picked)
"
```

預期輸出包含 `palette shape: (19, 3)`、`model loaded: WeatherSAM`、`batch keys: [...]`、`colorize ok: (8, 8, 3)`、`picked indices: {0: ..., 1: ..., 2: ..., 3: ...}`。

- [ ] **Step 4：commit**

```bash
git add segment-anything/scripts/eval/__init__.py segment-anything/scripts/eval/_eval_common.py
git commit -m "feat(eval): add shared eval utilities (model loader, val loader, palette)"
```

---

## Task 2：E1 — ACDC val 完整評估腳本

**Files:**
- Create: `segment-anything/scripts/eval/eval_e1_acdc_val_full.py`
- Output: `docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.md` + `.json`

- [ ] **Step 1：寫 E1 腳本**

```python
# segment-anything/scripts/eval/eval_e1_acdc_val_full.py
"""
E1: ACDC val 完整評估
產出：per-class × per-condition IoU 矩陣 + 整體 mIoU + per-condition mIoU
對應論文：Refign Tab.1 / CMA Tab.1
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (
    load_v15_model, build_acdc_val_loader, make_batched_input,
    CONDITION_NAMES, CITYSCAPES_CLASSES, OUTPUT_ROOT, DEFAULT_CKPT,
)

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
    model = load_v15_model(DEFAULT_CKPT, device=DEVICE)
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
            full = torch.full(
                (1, NUM_CLASSES, 256, 256), -10.0,
                device=DEVICE, dtype=low_res.dtype,
            )
            for k, c in enumerate(class_ids_out):
                full[0, c] = low_res[k]
            fused = model.context_fusion_head(full)                    # (1, 19, 256, 256)
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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / 'e1_acdc_val_results.json'
    json_data = {
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
    print(f'✅ JSON written: {json_path}')

    # ── 輸出 Markdown ──
    md_path = OUTPUT_ROOT / 'e1_acdc_val_results.md'
    lines = []
    lines.append('# E1: WeatherSAM v15 (E18) — ACDC val Evaluation')
    lines.append('')
    lines.append(f'**Checkpoint:** `{Path(DEFAULT_CKPT).name}`')
    lines.append(f'**Date:** 2026-05-14')
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
```

- [ ] **Step 2：執行 E1 並驗證輸出**

執行：
```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python segment-anything/scripts/eval/eval_e1_acdc_val_full.py 2>&1 | tail -10
```

預期：
- 完成時間 8–15 分鐘
- 終端最後輸出整體 mIoU 數值（接近 65%，允許 ±0.5% 隨機波動）
- 4 個條件的 mIoU 都有數字
- 產生 2 個檔案：
  - `docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.md`
  - `docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.json`

- [ ] **Step 3：人工驗證輸出格式正確**

```bash
head -30 docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.md
conda run -n sam_env python -c "
import json
with open('docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.json') as f:
    d = json.load(f)
assert 0.6 < d['overall_miou'] < 0.7, f\"miou {d['overall_miou']} out of expected range\"
assert set(d['per_condition_miou'].keys()) == {'fog','rain','snow','night'}
assert sum(d['sample_counts_by_condition'].values()) == d['num_samples_total']
print('JSON OK; overall_miou =', d['overall_miou'])
"
```

預期：`JSON OK` + mIoU 落在 0.6–0.7 範圍內。

- [ ] **Step 4：commit**

```bash
git add segment-anything/scripts/eval/eval_e1_acdc_val_full.py \
        docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.md \
        docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.json
git commit -m "feat(eval): add E1 ACDC val full evaluation + per-class/per-condition results"
```

---

## Task 3：E4 — 定性比較圖

**Files:**
- Create: `segment-anything/scripts/eval/viz_e4_qualitative.py`
- Output: `docs/experiments/v15-eval-2026-05-14/e4_qualitative.png`

- [ ] **Step 1：寫 E4 腳本**

```python
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
```

- [ ] **Step 2：執行 E4 並驗證輸出**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python segment-anything/scripts/eval/viz_e4_qualitative.py 2>&1 | tail -5
```

預期：終端輸出 `Picked indices: {0: ..., 1: ..., 2: ..., 3: ...}` 與 `Figure written: ...`，無 exception。

- [ ] **Step 3：人工驗證圖檔**

```bash
ls -la docs/experiments/v15-eval-2026-05-14/e4_qualitative.png
conda run -n sam_env python -c "
from PIL import Image
img = Image.open('docs/experiments/v15-eval-2026-05-14/e4_qualitative.png')
print('size:', img.size, 'mode:', img.mode)
assert img.size[0] > 1500 and img.size[1] > 2000, f'unexpected size {img.size}'
print('OK')
"
```

預期：圖檔約 2700×3600 像素（200 dpi × tight_layout），含 12 個 panel（4 條件 × 3 欄）。

- [ ] **Step 4：commit**

```bash
git add segment-anything/scripts/eval/viz_e4_qualitative.py \
        docs/experiments/v15-eval-2026-05-14/e4_qualitative.png
git commit -m "feat(eval): add E4 qualitative comparison figure (4 conditions x 3 cols)"
```

---

## Task 4：E5 — UAWarpC warp 與 confidence 可視化

**Files:**
- Create: `segment-anything/scripts/eval/viz_e5_warp_confidence.py`
- Output: `docs/experiments/v15-eval-2026-05-14/e5_warp_confidence.png`

- [ ] **Step 1：寫 E5 腳本**

```python
# segment-anything/scripts/eval/viz_e5_warp_confidence.py
"""
E5: UAWarpC warp 與 confidence 可視化
4 條件各 1 張樣本 × 4 欄：
    clear reference / warped reference / warped × confidence / adverse
對應論文：Refign Fig.7
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
    load_v15_model, pick_first_per_condition, denorm_image,
    CONDITION_NAMES, OUTPUT_ROOT, DEFAULT_CKPT, DEFAULT_VAL_CSV,
)

DEVICE = 'cuda'


def upsample_flow(flow_lr: torch.Tensor, out_hw: tuple) -> torch.Tensor:
    """把低解析度 flow (B, 2, H_lr, W_lr) 上採至 (B, 2, H_hi, W_hi)。
    flow 的數值是 pixel-offset 單位，所以同步乘上空間縮放係數。
    """
    B, _, H_lr, W_lr = flow_lr.shape
    H_hi, W_hi = out_hw
    flow_hi = F.interpolate(
        flow_lr, size=out_hw, mode='bilinear', align_corners=False,
    )
    sx = float(W_hi) / float(W_lr)
    sy = float(H_hi) / float(H_lr)
    flow_hi[:, 0] = flow_hi[:, 0] * sx
    flow_hi[:, 1] = flow_hi[:, 1] * sy
    return flow_hi


def main():
    from segment_anything.modeling.cma_utils import warp
    from utils.weather_dataloader import WeatherSegmentationDataset

    model = load_v15_model(DEFAULT_CKPT, device=DEVICE)
    ds = WeatherSegmentationDataset(
        csv_file=DEFAULT_VAL_CSV, image_size=1024, mode='val', force_raw_images=True,
    )
    picked = pick_first_per_condition(DEFAULT_VAL_CSV)
    print('Picked indices:', picked)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    column_titles = ['Clear Reference', 'Warped Reference',
                     'Warped × Confidence', 'Adverse (Target)']

    with torch.no_grad():
        for row, cid in enumerate(CONDITION_NAMES):
            idx = picked[cid]
            item = ds[idx]
            adverse = item['image'].unsqueeze(0).to(DEVICE)         # (1, 3, 1024, 1024)
            clear   = item['clear_image'].unsqueeze(0).to(DEVICE)

            # 1. 呼叫 pre_align 取得 flow + confidence（內部 out_size=(64, 64)）
            _ = model.fusion_module.pre_align(adverse, clear)
            flow_lr = model.fusion_module._last_flow.to(DEVICE)               # (1, 2, 64, 64)
            conf_lr = model.fusion_module._last_confidence_map.to(DEVICE)     # (1, 1, 64, 64)

            # 2. 上採 flow 至 1024×1024（同時 scaling pixel-offset 值）
            flow_hi = upsample_flow(flow_lr, out_hw=(1024, 1024))

            # 3. 用 flow 把 clear reference 影像 warp 至 adverse 視角
            warped_clear, valid = warp(clear, flow_hi, return_mask=True)      # (1, 3, 1024, 1024)

            # 4. 上採 confidence 至 1024×1024
            conf_hi = F.interpolate(conf_lr, size=(1024, 1024), mode='bilinear',
                                     align_corners=False)
            conf_hi = conf_hi.clamp(0.0, 1.0)                                   # (1, 1, 1024, 1024)
            conf_2d = conf_hi.squeeze(0).squeeze(0).cpu().numpy()              # (1024, 1024)

            # 5. warped_with_conf：低 confidence 區域 fade 為白色
            warped_np = denorm_image(warped_clear[0]).astype(np.float32)       # (H, W, 3) 0~255
            white = np.ones_like(warped_np) * 255.0
            alpha = conf_2d[..., None]                                          # (H, W, 1)
            warped_with_conf = (warped_np * alpha + white * (1.0 - alpha)).astype(np.uint8)

            # 渲染
            axes[row, 0].imshow(denorm_image(clear[0]))
            axes[row, 1].imshow(denorm_image(warped_clear[0]))
            axes[row, 2].imshow(warped_with_conf)
            axes[row, 3].imshow(denorm_image(adverse[0]))

            axes[row, 0].set_ylabel(
                CONDITION_NAMES[cid].capitalize(),
                fontsize=14, fontweight='bold',
            )
            for col in range(4):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / 'e5_warp_confidence.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Figure written: {out_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2：執行 E5 並驗證輸出**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python segment-anything/scripts/eval/viz_e5_warp_confidence.py 2>&1 | tail -5
```

預期：終端輸出 `Picked indices: ...` 與 `Figure written: ...`，無 exception。

- [ ] **Step 3：人工驗證圖檔**

```bash
conda run -n sam_env python -c "
from PIL import Image
img = Image.open('docs/experiments/v15-eval-2026-05-14/e5_warp_confidence.png')
print('size:', img.size, 'mode:', img.mode)
assert img.size[0] > 2500 and img.size[1] > 2500, f'unexpected size {img.size}'
print('OK')
"
```

預期：圖檔約 3200×3200 像素，含 16 個 panel（4 條件 × 4 欄）。視覺檢查：
- Warped Reference 與 Adverse 視角應對應同樣的道路結構
- Warped × Confidence 在動態物體（車、人）區域應該變白
- Clear Reference 看起來像晴天版的場景

- [ ] **Step 4：commit**

```bash
git add segment-anything/scripts/eval/viz_e5_warp_confidence.py \
        docs/experiments/v15-eval-2026-05-14/e5_warp_confidence.png
git commit -m "feat(eval): add E5 UAWarpC warp + confidence visualization (Refign Fig.7 style)"
```

---

## Task 5：最終彙整與彙報

- [ ] **Step 1：建立 README 索引彙整 3 個實驗的產出**

```bash
cat > docs/experiments/v15-eval-2026-05-14/README.md <<'EOF'
# v15 (E18) 權重評估實驗產出 — 2026-05-14

**Checkpoint:** `best_E18_mIoU65.06_LR4.6e-05.pth`

| 實驗 | 產出 | 對應論文 |
|------|------|----------|
| E1 — ACDC val 完整評估 | [`e1_acdc_val_results.md`](e1_acdc_val_results.md), [`e1_acdc_val_results.json`](e1_acdc_val_results.json) | Refign Tab.1 / CMA Tab.1 |
| E4 — 定性比較圖 | [`e4_qualitative.png`](e4_qualitative.png) | Refign Fig.4 / CMA Fig.4 |
| E5 — UAWarpC warp + confidence | [`e5_warp_confidence.png`](e5_warp_confidence.png) | Refign Fig.7 |

## 重現方式

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python segment-anything/scripts/eval/eval_e1_acdc_val_full.py
conda run -n sam_env python segment-anything/scripts/eval/viz_e4_qualitative.py
conda run -n sam_env python segment-anything/scripts/eval/viz_e5_warp_confidence.py
```
EOF
```

- [ ] **Step 2：commit README**

```bash
git add docs/experiments/v15-eval-2026-05-14/README.md
git commit -m "docs(eval): add README index for v15 eval experiments"
```

- [ ] **Step 3：終端彙報**

執行：
```bash
echo "=== 產出檔案 ==="
ls -lh docs/experiments/v15-eval-2026-05-14/
echo ""
echo "=== Overall mIoU 與 per-condition mIoU ==="
conda run -n sam_env python -c "
import json
with open('docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.json') as f:
    d = json.load(f)
print(f'Overall mIoU: {d[\"overall_miou\"]*100:.2f}%')
for c, v in d['per_condition_miou'].items():
    print(f'  {c:6s}: {v*100:.2f}%')
"
```

預期：列出 6 個檔案、整體 mIoU 與 4 條件 mIoU。

---

## 風險與 fallback

| 風險 | 偵測方式 | Fallback |
|------|----------|----------|
| Checkpoint 載入有大量 missing keys（>10）| `load_v15_model` print 訊息 | 改 strict=True 並補上缺鍵 |
| ACDC val 跑出的 mIoU 偏離 65% 太多（差 > 2%）| Task 2 Step 3 JSON 驗證 assert | 檢查 inference 流程是否完全對齊 trainer validate_epoch |
| Flow upsample scaling 比例錯誤造成 warp 結果偏移 | Task 4 Step 3 人工檢視圖 | 嘗試移除 sx/sy 倍率（flow 可能已是正規化單位）|
| E5 confidence map 全部 ≈ 1.0（沒區分動靜態物體）| Task 4 Step 3 視覺檢查 | 檢查 `_last_confidence_map` 在 pre_align 後立即讀取，未被下一次 forward 覆蓋 |

---

## 範圍外（下一個 plan 才做）

- E2：head-to-head vs CMA（需設定 CMA conda env，獨立子任務）
- E3：ACDC test set 提交（需另寫 inference + 上傳腳本）
- 跨資料集泛化（Dark Zurich、ACG benchmark 等）
- 消融實驗（需重訓）
