"""
ACDC CSV 資料驗證腳本
驗證 ref_complete.csv 中的每筆資料是否符合 WeatherSAM 訓練架構的要求：
  1. image_path     → 3-channel RGB PNG（輸入影像）
  2. ref_mask_path  → 3-channel color PNG，且非純黑（MaskEncoder 輸入）
  3. gt_path        → 灰階 PNG，pixel 值 ∈ {0..18, 255}（CrossEntropyLoss target）
  4. lat / lon      → 數值欄位存在
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CSV_PATH = "/home/rvl1421/SAM_research-1/Datasets/ref_complete.csv"
VALID_CLASS_IDS = set(range(19)) | {255}  # 0-18 + ignore index 255

# ── 讀取 CSV ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"總筆數: {len(df)}")
print(f"欄位: {list(df.columns)}\n")

# ── 檢查必要欄位 ─────────────────────────────────────────────────────────────
required_cols = ["image_path", "ref_mask_path", "gt_path", "lat", "lon"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"[ERROR] 缺少欄位: {missing}")
    sys.exit(1)
print("[OK] 必要欄位齊全\n")

# ── 逐筆驗證 ─────────────────────────────────────────────────────────────────
errors = []

def add_error(idx, col, msg):
    errors.append({"row": idx, "col": col, "msg": msg})

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
    # ---------- image_path ----------
    img_p = row["image_path"]
    if not os.path.exists(img_p):
        add_error(idx, "image_path", f"檔案不存在: {img_p}")
    else:
        try:
            img = Image.open(img_p).convert("RGB")
            arr = np.array(img)
            if arr.ndim != 3 or arr.shape[2] != 3:
                add_error(idx, "image_path", f"非 3-channel: shape={arr.shape}")
        except Exception as e:
            add_error(idx, "image_path", f"讀取失敗: {e}")

    # ---------- ref_mask_path ----------
    ref_p = row["ref_mask_path"]
    if not os.path.exists(ref_p):
        add_error(idx, "ref_mask_path", f"檔案不存在: {ref_p}")
    else:
        try:
            ref = Image.open(ref_p).convert("RGB")
            arr = np.array(ref)
            if arr.ndim != 3 or arr.shape[2] != 3:
                add_error(idx, "ref_mask_path", f"非 3-channel: shape={arr.shape}")
            elif arr.sum() == 0:
                add_error(idx, "ref_mask_path", "影像全黑（void），MaskEncoder 無有效輸入")
        except Exception as e:
            add_error(idx, "ref_mask_path", f"讀取失敗: {e}")

    # ---------- gt_path ----------
    gt_p = row["gt_path"]
    if not os.path.exists(gt_p):
        add_error(idx, "gt_path", f"檔案不存在: {gt_p}")
    else:
        try:
            gt = Image.open(gt_p)
            arr = np.array(gt)
            unique_vals = set(arr.flatten().tolist())
            invalid_vals = unique_vals - VALID_CLASS_IDS
            if invalid_vals:
                add_error(idx, "gt_path", f"包含非法 class ID: {invalid_vals}")
        except Exception as e:
            add_error(idx, "gt_path", f"讀取失敗: {e}")

    # ---------- lat / lon ----------
    if pd.isna(row["lat"]) or pd.isna(row["lon"]):
        add_error(idx, "lat/lon", "lat 或 lon 為 NaN")

# ── 結果報告 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if not errors:
    print(f"[PASS] 所有 {len(df)} 筆資料驗證通過，無任何錯誤。")
else:
    err_df = pd.DataFrame(errors)
    print(f"[FAIL] 發現 {len(errors)} 筆錯誤（共 {len(df)} 筆）：\n")
    # 彙整各欄位錯誤數量
    print(err_df.groupby("col")["msg"].count().rename("錯誤數").to_string())
    print("\n--- 前 20 筆詳細錯誤 ---")
    print(err_df.head(20).to_string(index=False))

    # 存成 CSV 方便後續檢查
    err_out = "/home/rvl1421/SAM_research-1/Datasets/acdc_validation_errors.csv"
    err_df.to_csv(err_out, index=False)
    print(f"\n完整錯誤清單已儲存至: {err_out}")

print("="*60)
