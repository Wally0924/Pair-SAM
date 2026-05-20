# Spec: test_inference.py 改寫為 v15 架構推論版

**Date:** 2026-05-20
**Target file:** `segment-anything/test_inference.py`
**Target checkpoint:** `segment-anything/outputs_weather_sam_mask2former_testv15/best_E27_mIoU65.68_LR4.0e-05.pth`

## 目的

將 `test_inference.py` 從舊版（testv7、CMAAlignment diagnostic + temperature sweep + 可選 zero-embedding ablation + postprocess_masks 上採至 original_size）改寫為 **與 v15 eval pipeline（`scripts/eval/_eval_common.py` + `eval_e1_acdc_val_full.py`）完全對齊**的「per-image 可視化 + per-image mIoU + 最終 metrics 彙總」工具。

## 範圍

- **In scope**：模型載入、dataloader、forward + post-processing、可視化、metrics 彙總、CLI 主程式。
- **Out of scope**：confusion-matrix 評估表格（已由 E1 script 負責）、CMA alignment 診斷（保留給未來專用 script）、temperature ablation。

## 與舊版的關鍵差異

| 面向 | 舊版（testv7） | 新版（v15） |
|---|---|---|
| Checkpoint 載入 | `build_weather_sam_vit_h(checkpoint=...)` 內部 strict load | `load_v15_model()`：`build_*(checkpoint=None)` → `torch.load` → `load_state_dict(strict=False)` → `enable_vgg_adapter('pre')` |
| Dataset | 手動設 `test_ds.has_cached_features=False` | 建構時 `force_raw_images=True` |
| batched_input | 包含 `clear_embedding`/`image_embedding`/`invalid_mask`，分支處理 raw vs cached | 僅 `image`/`clear_image`/`text_prompts`/`original_size`/`condition_id` |
| 後處理上採樣 | `model.postprocess_masks(input=(1024,1024), original=(H,W))` | `F.interpolate(fused, (1024,1024), bilinear)` |
| 評估解析度 | original_size (H_orig×W_orig) | 1024×1024（與 trainer/E1 對齊） |
| GT 對齊 | 縮到 pred shape；invalid_mask 設 255 | 縮到 1024；invalid_mask 同步縮到 1024 後設 255 |
| Reference ablation (`use_reference=False`) | 用 zeros(256,64,64) 取代 clear_embedding | **移除**（v15 把 clear_image 當必要輸入） |
| Temperature sweep | softmax(/T).argmax | **移除**（argmax 等價於 T=1.0） |
| CMA diagnostic hook | `register_diagnostic_hooks` 印 f_curr/f_ref/cosine | **移除** |

## 模組設計

### 依賴 import

```python
import sys
from pathlib import Path
# 將 scripts/eval 加入 sys.path 以複用 _eval_common
_SEGANY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SEGANY_ROOT / 'scripts' / 'eval'))
from _eval_common import (
    load_v15_model, build_acdc_val_loader, make_batched_input,
    CITYSCAPES_CLASSES, CITYSCAPES_PALETTE, CONDITION_NAMES,
)
```

避免在 `test_inference.py` 重複實作模型載入 / dataloader 邏輯。

### InferenceRunner

```python
class InferenceRunner:
    def __init__(self, model, device, output_dir): ...
    @torch.no_grad()
    def predict_single_image(self, batch) -> np.ndarray:
        """回傳 (1024,1024) 的 pred mask（int64）。"""
    def prepare_gt_1024(self, batch) -> Optional[np.ndarray]:
        """回傳 (1024,1024) GT，invalid 區域設 255；若 batch 無 gt_mask 則 None。"""
    def visualize(self, batch, pred, gt, idx, miou): ...
    def run_inference(self, loader, num_samples=None): ...
```

#### `predict_single_image` 內部步驟

```
batched_input = make_batched_input(batch, device)
outputs = model(batched_input)
low_res = outputs[0]['low_res_logits'].squeeze(0)   # (K,256,256)
class_ids = outputs[0]['class_ids']                  # List[int]
full = torch.full((1, 19, 256, 256), -10.0, device=device, dtype=low_res.dtype)
for k, c in enumerate(class_ids):
    full[0, c] = low_res[k]
fused    = model.context_fusion_head(full)           # (1,19,256,256)
fused_hr = F.interpolate(fused, (1024,1024), mode='bilinear', align_corners=False)
pred     = fused_hr.argmax(1).squeeze(0)             # (1024,1024)
return pred.cpu().numpy()
```

> 與 `scripts/eval/eval_e1_acdc_val_full.py` 邏輯 1:1 對齊。

#### `prepare_gt_1024`

```
gt = batch['gt_mask'][0].to(device).long()            # (H_orig, W_orig)
gt_1024 = F.interpolate(gt[None,None].float(), (1024,1024), mode='nearest').long()[0,0]
inv = batch['invalid_mask'][0].to(device)             # (H_orig, W_orig) bool
inv_1024 = F.interpolate(inv[None,None].float(), (1024,1024), mode='nearest').bool()[0,0]
gt_1024[inv_1024] = 255
return gt_1024.cpu().numpy()
```

#### `visualize`

- Figure：1024 顯示空間（圖檔不需要原始解析度）。
- Input image：從 `batch['image'][0]` (3,1024,1024) 直接取，不再 resize。
- Clear reference RGB：從 `batch['reference_mask'][0]` (3,H,W)，必要時 resize 到 1024。
- Pred / GT：colorize_19class 後直接顯示。
- 標題：`Sample idx | Image mIoU: x.xxxx`。
- 圖例：`pred + gt` 出現的 class id（GT 若 None 則只看 pred）。

`colorize_19class` 可直接 import `_eval_common`，不需要在本檔重新定義。

#### `run_inference`

```
metrics = SegMetricsCalculator(classes=CITYSCAPES_CLASSES)
for i, batch in enumerate(pbar):
    pred = self.predict_single_image(batch)
    gt   = self.prepare_gt_1024(batch) if 'gt_mask' in batch else None
    if gt is not None:
        miou = SegMetricsCalculator.compute_image_miou(pred, gt, 19)
        condition = ...  # batch['condition'][0] 若存在則傳入
        metrics.update(pred, gt, condition=condition)
        print(f"📊 Image {i:03d} | mIoU: {miou:.4f}")
    self.visualize(batch, pred, gt, idx=i, miou=miou if gt is not None else None)
    if num_samples is not None and i+1 >= num_samples: break
metrics.print_report(metrics.compute())
```

### `__main__`

```python
CHECKPOINT_PATH = ".../outputs_weather_sam_mask2former_testv15/best_E27_mIoU65.68_LR4.0e-05.pth"
TEST_CSV_PATH   = ".../Datasets/acdc_adverse_ref_rgb_val.csv"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR      = "inference_viz_acdc_v15_E27"

model  = load_v15_model(CHECKPOINT_PATH, device=DEVICE)
loader = build_acdc_val_loader(TEST_CSV_PATH, batch_size=1, num_workers=4)

runner = InferenceRunner(model, DEVICE, OUTPUT_DIR)
runner.run_inference(loader, num_samples=None)
```

## 成功標準

1. 腳本能在 sam_env 內無錯執行，產出 `inference_viz_acdc_v15_E27/result_XXX.png`。
2. 跑完整 ACDC val（106 張）後，`metrics.print_report` 印出的 overall mIoU 應落在 ~65.5–65.9%（與 E1 結果同口徑、允許因 per-image mIoU 平均 vs confusion-matrix nanmean 的些微差異）。
3. 程式碼從 ~375 行壓到 ~180 行；沒有未使用的 import、沒有 zero-embedding / temperature / hook 殘留。
4. 與 `scripts/eval/eval_e1_acdc_val_full.py` 的 forward+post-process 區段邏輯 1:1 對齊。

## 風險與緩解

- **`sys.path` 插入 scripts/eval**：若未來 `_eval_common` 被搬移，本檔需同步調整 path。風險低（檔案近 commit；已成為 v15 eval 共用工具）。
- **`SegMetricsCalculator.update(...,condition=...)`**：需確認該方法簽章支援 condition 參數（舊檔已在用，預期相容；實作前先讀一次 `utils/seg_metrics.py` 確認）。
- **`reference_mask`**：dataloader 可能回傳 (3, H_orig, W_orig)；可視化需注意要 resize 到 1024 才能與 input 並排。

## 文件對齊

本 spec 完成後進入 `superpowers:writing-plans` 撰寫實作計畫，再由 `superpowers:subagent-driven-development` 或直接實作執行。
