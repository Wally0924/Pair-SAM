# v15 權重立即評估實驗 — E1 / E4 / E5

**日期：** 2026-05-14
**分支：** feat/image-pair-fusion
**動機：** 使用既有 `best_E18_mIoU65.06_LR4.6e-05.pth` 權重（無需重訓），產出三項可立即與 Refign / CMA 論文對照的實驗數據與視覺化資產，供 NCAR2026 paper §4 使用。

---

## 1. 範圍與不在範圍內

### 範圍內

| 編號 | 實驗 | 對應論文 | 產出 |
|------|------|----------|------|
| **E1** | ACDC val 完整評估（per-class × per-condition）| Refign Tab.1, CMA Tab.1 | 表格 + JSON |
| **E4** | 定性比較圖（input / 我們的 pred / GT × 4 條件）| Refign Fig.4, CMA Fig.4 | PNG figure |
| **E5** | UAWarpC warp 與 confidence 可視化 | Refign Fig.7 | PNG figure |

### 不在本次範圍內

- 重新訓練或 fine-tune（已明確排除）
- 與 CMA / Refign 的 head-to-head（需設定外部 repo，獨立子任務）
- ACDC test set 提交（獨立子任務）
- 消融實驗（component on/off）

---

## 2. 共用架構

### 檔案佈局

```
segment-anything/scripts/eval/
├── _eval_common.py              # 共用工具：load_v15, build_val_loader, palette
├── eval_e1_acdc_val_full.py     # E1
├── viz_e4_qualitative.py        # E4
└── viz_e5_warp_confidence.py    # E5
```

### 共用元件 (`_eval_common.py`)

```python
DEFAULT_CKPT  = "outputs_weather_sam_mask2former_testv15/best_E18_mIoU65.06_LR4.6e-05.pth"
DEFAULT_CSV   = "../Datasets/acdc_adverse_ref_rgb_val.csv"
OUTPUT_DIR    = "docs/experiments/v15-eval-2026-05-14"
CONDITION_NAMES   = {0: 'fog', 1: 'rain', 2: 'snow', 3: 'night'}
CITYSCAPES_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic light', 'traffic sign', 'vegetation', 'terrain',
                      'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                      'motorcycle', 'bicycle']  # 19 類，與 ACDC trainIds 對齊

def load_v15_model(ckpt_path: str, device: str = 'cuda'):
    """載入 best_E18 並啟用 v5 cross-attn pre-hook adapter。"""
    model = build_weather_sam_vit_h(num_classes=19, checkpoint=None)
    state = torch.load(ckpt_path, map_location='cpu')
    # state 可能直接是 state_dict 或 {'model': state_dict, 'epoch': ...}
    sd = state.get('model_state_dict', state.get('model', state))
    model.load_state_dict(sd, strict=False)  # strict=False 容忍訓練時新增的 buffer
    model.enable_vgg_adapter('pre')
    return model.to(device).eval()

def build_acdc_val_loader(csv_path: str, batch_size: int = 1):
    """重用既有的 WeatherSegmentationDataset，但 batch_size=1 簡化 per-sample logging。"""
    from utils.weather_dataloader import WeatherSegmentationDataset
    ds = WeatherSegmentationDataset(csv_path, split='val', image_size=1024)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

def colorize_19class(mask_np: np.ndarray) -> np.ndarray:
    """19-class trainID mask → RGB，與 ACDC 官方一致。直接從 test_inference.py 複製 CITYSCAPES_PALETTE。"""
    ...
```

**設計理由：**
- 採取 *standalone scripts* 而非修改 `weather_trainer.py`：read-only 評估，不影響訓練可重現性
- `strict=False` 載入：v5 加了 `_last_kv_keep_ratio` 等新 buffer，舊 checkpoint 沒有；strict=True 會無謂報錯
- batch_size=1：方便逐張記錄 condition_id 而不需要拆 batch

---

## 3. E1：ACDC val 完整評估

### 目標

對 ACDC val 的 406 張影像跑一次 inference，產出：
1. **整體 mIoU**（19 類平均）
2. **per-class IoU** × 19 類
3. **per-condition mIoU**（fog / rain / snow / night 各一）
4. **per-condition × per-class IoU 矩陣**（4 × 19）

直接對應 Refign Tab.1 與 CMA Tab.1 的 mIoU 報告格式。

### 演算法

對每張 sample：
1. 從 batch dict 取出 `condition_id` (int) 與 `gt_mask`
2. 模型 forward → `fused_logits_hr` (1, 19, 1024, 1024)
3. argmax → `pred_cls` (1024, 1024)
4. 過濾 `valid = (gt != 255)`
5. 更新 5 個 confusion matrix：`overall` + `cond_0` + `cond_1` + `cond_2` + `cond_3`
   - 用 `torch.bincount(gt * 19 + pred, minlength=19*19).reshape(19, 19)` 累積
6. 全部跑完後，對每個 confusion matrix 計算 per-class IoU：
   - `IoU_c = diag[c] / (row[c] + col[c] - diag[c])`
   - `mIoU = mean(IoU_c for c in present_classes)`

### 輸出格式

`docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.md`：

```markdown
# E1: WeatherSAM v15 (E18) — ACDC val Evaluation

**Checkpoint:** best_E18_mIoU65.06_LR4.6e-05.pth
**Date:** 2026-05-14
**Samples:** 406 (fog=100, rain=100, snow=100, night=106)
**Overall mIoU:** 65.06%

## Per-Condition mIoU

| Condition | mIoU |
|-----------|------|
| Fog       | 71.xx |
| Rain      | 68.xx |
| Snow      | 64.xx |
| Night     | 56.xx |
| **All**   | **65.06** |

## Per-Class × Per-Condition IoU (%)

| Class       | Fog | Rain | Snow | Night | All |
|-------------|-----|------|------|-------|-----|
| road        | ... | ...  | ...  | ...   | ... |
| sidewalk    | ... | ...  | ...  | ...   | ... |
| ... (19 rows total)
```

對應 JSON：
```json
{
  "checkpoint": "best_E18_mIoU65.06_LR4.6e-05.pth",
  "overall_miou": 0.6506,
  "per_condition_miou": {"fog": 0.71, "rain": 0.68, "snow": 0.64, "night": 0.56},
  "per_class_iou": {"road": [fog, rain, snow, night, all], ...},
  "sample_counts": {"fog": 100, "rain": 100, "snow": 100, "night": 106}
}
```

### 注意事項

- ACDC val 的 condition 分布：fog/rain/snow 各 100 張、night 106 張（依 csv 內容）— 腳本應自動計數而非寫死
- `gt == 255` 像素必須排除（invalid mask 區域）
- 該腳本應在 ~10 分鐘內完成 406 張 inference（取決於 GPU 使用率）

---

## 4. E4：定性比較圖

### 目標

選 4 張代表性 ACDC val 樣本（fog/rain/snow/night 各 1 張），並排呈現：
- **Column 1:** 惡劣天氣輸入圖
- **Column 2:** 我們的預測（19-class 上色 mask）
- **Column 3:** Ground Truth（19-class 上色 mask）

格式對應 Refign Fig.4（缺第 4 欄的 baseline 比較，因為本次不做 E2）。

### 樣本選擇

採取 *固定樣本* 而非自動挑選，確保可重現：

| Condition | Sample（從 acdc_adverse_ref_rgb_val.csv 的 image_path 末段） |
|-----------|----------------------------------------------------------|
| fog       | `GOPR0476_frame_000761_rgb_anon.png` |
| rain      | `GOPR0400_frame_000xxx_rgb_anon.png`（實作時從 val csv 取第一張 rain）|
| snow      | `GOPR0xxx_frame_000xxx_rgb_anon.png`（第一張 snow）|
| night     | `GOPR0xxx_frame_000xxx_rgb_anon.png`（第一張 night）|

腳本實作時用 condition_id 過濾 csv 取每類首張，保證可重現且不依賴特定檔名硬編碼。

### 演算法

1. 從 val csv 過濾出 condition_id ∈ {0, 1, 2, 3} 各取首張
2. 對 4 張 sample 跑 inference → pred_cls
3. matplotlib 排版：
   ```
   subplot grid: 4 rows × 3 columns
   row labels: "Fog" / "Rain" / "Snow" / "Night"
   column titles: "Input" / "WeatherSAM (Ours)" / "Ground Truth"
   ```
4. 上色：`colorize_19class(mask)` 使用 Cityscapes 標準調色盤
5. 儲存：`docs/experiments/v15-eval-2026-05-14/e4_qualitative.png` @ 200 dpi

### 圖檔規格

- 解析度：每子圖 256×128（顯示），整體 16:9 比例
- 字體：Arial 12pt for titles
- 背景：白底
- 邊距：0.05 inch tight_layout

---

## 5. E5：UAWarpC warp 與 confidence 可視化

### 目標

對同 4 張樣本（與 E4 一致），呈現 UAWarpC 對齊模組的內部行為：
- **Column 1:** 晴天參考圖（clear reference）
- **Column 2:** 用 flow 把參考圖 warp 至 adverse 視角後的結果
- **Column 3:** Warped reference × confidence mask（低信心區域 fade 為白色）
- **Column 4:** 惡劣天氣輸入圖

對應 Refign Fig.7，能視覺化 alignment 品質與 confidence 估測的合理性。

### 演算法

對每張 sample：
1. 載入 `adverse_img`（1024×1024 RGB）與 `clear_ref_img`（1024×1024 RGB）
2. 呼叫 `model.fusion_module.pre_align(adverse, clear_ref, out_size=(64, 64))`
3. 從 `fusion_module._last_flow` 取得 flow 場（B, 2, 64, 64），上採至 (1024, 1024)
4. 從 `fusion_module._last_confidence_map` 取得 confidence (B, 1, 64, 64)，上採至 (1024, 1024)
5. 用 `cma_utils.warp(clear_ref_tensor, flow_upscaled)` 把參考圖 warp 至 adverse 視角
6. 計算 `warped_with_conf = warped_ref * conf + (1 - conf) * 1.0`（低信心區白色）
7. matplotlib 4×4 排版（4 條件 × 4 欄）

### 圖檔規格

- 解析度：每子圖 256×256，整體 (4 × 256) × (4 × 256) = 1024 × 1024 大圖
- 對應 Refign Fig.7 的視覺風格（confidence map 用灰度疊白底）
- 儲存：`docs/experiments/v15-eval-2026-05-14/e5_warp_confidence.png` @ 200 dpi

### 技術注意

- `pre_align` 內部 `out_size=(64, 64)` 是 ViT-H 的 token grid，flow 在這個解析度產生
- 把 flow 上採至 1024×1024 時需 **同時** scaling flow vector 的數值（因為 flow 是 pixel-offset 單位）
- `confidence_map` 是 [0, 1] 之間的 float，可直接用作 alpha mask
- `_last_flow` 與 `_last_confidence_map` 在 `pre_align` 結尾才 `.cpu()`，腳本必須在每張處理完立即讀取，否則會被下一張覆蓋

---

## 6. 成功驗證標準

### E1

- [ ] `e1_acdc_val_results.md` 產生，整體 mIoU 落在 64.5–65.5%（與訓練 log 的 65.06% 接近，允許 ±0.5% 隨機波動）
- [ ] 4 個 condition 都有 ≥ 50 樣本參與計算
- [ ] 19 類 IoU 全部有數值（即使是 0），JSON 與 markdown 表格一致

### E4

- [ ] PNG 檔產生，4 × 3 grid 佈局清晰
- [ ] 4 條件各 1 張樣本，顏色與 ACDC 官方 GT 一致
- [ ] 標題清楚標示「WeatherSAM (Ours)」

### E5

- [ ] PNG 檔產生，4 × 4 grid 佈局清晰
- [ ] Warped reference 與 adverse 視角對齊（視覺上能對應主要結構）
- [ ] Confidence mask 在動態物體（車輛、行人）區域明顯偏低（符合 UAWarpC 預期行為）

---

## 7. 變更摘要

| 檔案 | 動作 | 性質 |
|------|------|------|
| `segment-anything/scripts/eval/_eval_common.py` | 新增 | 共用工具 |
| `segment-anything/scripts/eval/eval_e1_acdc_val_full.py` | 新增 | E1 評估腳本 |
| `segment-anything/scripts/eval/viz_e4_qualitative.py` | 新增 | E4 視覺化 |
| `segment-anything/scripts/eval/viz_e5_warp_confidence.py` | 新增 | E5 視覺化 |
| `docs/experiments/v15-eval-2026-05-14/` | 新增資料夾 | 實驗產出 |
| 訓練程式碼 (`weather_trainer.py`, `train.py`) | **不動** | 隔離 |
| 模型程式碼 (`vgg_adapter.py`, `weather_sam.py`) | **不動** | 隔離 |

---

## 8. 預期執行時間

| 步驟 | 預估 |
|------|------|
| `_eval_common.py` + E1 腳本撰寫 | 1.5 小時 |
| E1 跑 inference（406 張）| 10–15 分鐘 |
| E4 腳本撰寫 + 跑（4 張）| 1 小時 |
| E5 腳本撰寫 + 跑（4 張）| 1.5 小時 |
| **總計** | **約 4.5 小時** |
