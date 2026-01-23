import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys

# 定義你的資料集類別範圍 (根據 new_loss.py)
# 0: road ... 18: bicycle
VALID_CLASS_MIN = 0
VALID_CLASS_MAX = 18
IGNORE_INDEX = 255  # 常見的忽略標籤

def check_dataset_integrity(csv_path, base_dir=None):
    print(f"🔍 開始檢查資料集清單: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ 錯誤: 找不到 CSV 檔案 {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    total_files = len(df)
    print(f"📦 總樣本數: {total_files}")
    
    error_log = []
    
    # 初始化統計
    warnings_ref_mask_dark = 0
    errors_file_missing = 0
    errors_gt_invalid = 0
    errors_shape_mismatch = 0

    # 進度條
    pbar = tqdm(total=total_files, desc="Checking Integrity", unit="img")
    
    for idx, row in df.iterrows():
        # 1. 取得路徑 (若有 base_dir 則串接)
        img_path = row['image_path']
        gt_path = row['gt_path']
        ref_path = row.get('ref_mask_path', None) # Ref mask 可能不一定每張都有，視你的設計而定
        
        if base_dir:
            img_path = os.path.join(base_dir, img_path)
            gt_path = os.path.join(base_dir, gt_path)
            if ref_path: ref_path = os.path.join(base_dir, ref_path)

        # ------------------------------------------------------
        # 2. 檢查檔案是否存在
        # ------------------------------------------------------
        if not os.path.exists(img_path):
            error_log.append(f"[Missing] Row {idx}: Image not found at {img_path}")
            errors_file_missing += 1
            pbar.update(1)
            continue
            
        if not os.path.exists(gt_path):
            error_log.append(f"[Missing] Row {idx}: GT not found at {gt_path}")
            errors_file_missing += 1
            pbar.update(1)
            continue
            
        if ref_path and not os.path.exists(ref_path):
            error_log.append(f"[Missing] Row {idx}: Ref Mask not found at {ref_path}")
            errors_file_missing += 1
            pbar.update(1)
            continue

        # ------------------------------------------------------
        # 3. 讀取影像並檢查
        # ------------------------------------------------------
        # 讀取 GT (Grayscale)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            error_log.append(f"[ReadFail] Row {idx}: Failed to read GT {gt_path}")
            pbar.update(1)
            continue
            
        # 檢查 GT 數值範圍
        unique_ids = np.unique(gt)
        # 過濾掉 255
        valid_ids = unique_ids[unique_ids != IGNORE_INDEX]
        
        if len(valid_ids) > 0:
            if valid_ids.max() > VALID_CLASS_MAX or valid_ids.min() < VALID_CLASS_MIN:
                error_log.append(f"[InvalidID] Row {idx}: GT contains invalid IDs {valid_ids[valid_ids > VALID_CLASS_MAX]}")
                errors_gt_invalid += 1

        # ------------------------------------------------------
        # 4. 關鍵檢查: Reference Mask 格式
        # ------------------------------------------------------
        if ref_path:
            ref = cv2.imread(ref_path) # BGR default
            
            if ref is None:
                error_log.append(f"[ReadFail] Row {idx}: Failed to read Ref Mask {ref_path}")
            else:
                # [Critical Risk Check] 檢查是否為 Class ID Map 偽裝成 RGB
                # 如果最大像素值很小 (例如 <= 18)，且是一個 3 通道圖片
                # 這意味著它可能是存成 PNG 的 ID Map，而不是視覺化後的 RGB 圖
                ref_max = ref.max()
                
                if ref_max <= 18 and ref_max > 0:
                    # 這是一個強烈警告
                    msg = f"[RefMaskTooDark] Row {idx}: Ref Mask max value is {ref_max}. It looks like an ID map, NOT an RGB visualization. Model will see this as black."
                    # 我們只記錄前 5 個這類錯誤，避免 log 爆炸
                    if warnings_ref_mask_dark < 5:
                        error_log.append(msg)
                    warnings_ref_mask_dark += 1

                # 檢查維度一致性 (簡單檢查長寬比是否大致相符，不要求像素級別完全一樣，因為 DataLoader 會 Resize)
                # 這裡嚴格檢查形狀是否匹配 GT
                if ref.shape[:2] != gt.shape:
                    error_log.append(f"[ShapeMismatch] Row {idx}: GT {gt.shape} vs Ref {ref.shape[:2]}")
                    errors_shape_mismatch += 1

        pbar.update(1)

    pbar.close()
    
    # ------------------------------------------------------
    # 5. 輸出報告
    # ------------------------------------------------------
    print("\n" + "="*50)
    print("📊 資料集檢查報告 (Dataset Integrity Report)")
    print("="*50)
    
    if len(error_log) == 0 and warnings_ref_mask_dark == 0:
        print("✅ 完美！資料集看起來非常健康。")
        print("   Ready to train! 🚀")
        return True
    
    print(f"❌ 發現檔案遺失: {errors_file_missing}")
    print(f"❌ 發現無效類別 ID: {errors_gt_invalid}")
    print(f"❌ 發現形狀不匹配: {errors_shape_mismatch}")
    
    if warnings_ref_mask_dark > 0:
        print(f"⚠️ [嚴重警告] 發現 {warnings_ref_mask_dark} 張 Reference Mask 數值過低 (<=18)！")
        print("   -> 請確認你的 Ref Mask 是 'RGB Visualization' 而不是 'Class ID Map'。")
        print("   -> 在 weather_sam.py 中，代碼會執行 `mask / 255.0`。")
        print("   -> 如果輸入最大值只有 18，除以 255 後會接近 0，導致 Mask Encoder 失效。")
    
    print("\n詳細錯誤日誌 (前 20 筆):")
    for msg in error_log[:20]:
        print("   " + msg)
        
    if len(error_log) > 20:
        print(f"   ... (還有 {len(error_log) - 20} 筆錯誤)")

    # 寫入 Log 檔
    with open("data_integrity_log.txt", "w") as f:
        for line in error_log:
            f.write(line + "\n")
    print("\n📄 完整錯誤已寫入 data_integrity_log.txt")
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/home/rvl1421/SAM_research/Datasets/train_cached.csv", help="Path to your train.csv or val.csv")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory if CSV paths are relative")
    args = parser.parse_args()
    
    check_dataset_integrity(args.csv, args.base_dir)