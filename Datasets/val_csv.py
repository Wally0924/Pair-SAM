import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= 設定 =================
# 你要檢查的 CSV 檔案列表
CSV_FILES = [
    "train_all.csv",
    "val_all.csv"
    # "Datasets/test.csv" # Test集通常沒有 GT，可以先註解掉或另外處理
]

# 隨機抽樣檢查的數量 (設為 0 代表檢查全部，建議先設 5 張看圖，再設 0 跑全檢)
NUM_VISUALIZE = 5 
# =======================================

def check_csv(csv_path):
    print(f"\n🔍 正在檢查: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 檔案: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"   總筆數: {len(df)}")
    
    # 統計錯誤
    missing_files = 0
    mismatched_names = 0
    invalid_labels = 0
    
    # 隨機抽樣用於視覺化
    sample_indices = np.random.choice(len(df), min(NUM_VISUALIZE, len(df)), replace=False)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        img_path = row['image_path']
        ref_path = row['ref_mask_path']
        gt_path = row.get('gt_path', None) # Test set 可能沒有 GT

        # 1. 檢查檔案是否存在
        if not os.path.exists(img_path):
            print(f"❌ 遺失影像: {img_path}")
            missing_files += 1
            continue
        
        if pd.notna(ref_path) and not os.path.exists(ref_path):
            print(f"❌ 遺失參考圖: {ref_path}")
            missing_files += 1
        
        if pd.notna(gt_path) and not os.path.exists(gt_path):
            print(f"❌ 遺失標籤: {gt_path}")
            missing_files += 1

        # 2. 檢查檔名核心 ID 是否一致 (防止 A圖 配 B標籤)
        # img: frankfurt_000000_000294_leftImg8bit_foggy_beta_0.02.png
        # gt:  frankfurt_000000_000294_gtFine_labelTrainIds.png
        img_core = os.path.basename(img_path).split('_leftImg8bit')[0]
        
        if pd.notna(gt_path):
            gt_core = os.path.basename(gt_path).split('_gtFine')[0]
            if img_core != gt_core:
                print(f"⚠️ 檔名不匹配 (Line {idx}):\n   Img: {img_core}\n   GT:  {gt_core}")
                mismatched_names += 1

        # 3. 檢查圖像內容 (只對部分或有問題的進行讀取，避免太慢)
        # 這裡我們讀取每一張圖的 Header 來檢查，只有視覺化時才完整解碼
        if idx in sample_indices:
            visualize_sample(img_path, ref_path, gt_path, idx)
        
        # 4. 針對 GT 檢查數值範圍 (確保轉換腳本有執行)
        if pd.notna(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is None:
                print(f"❌ 無法讀取 GT: {gt_path}")
                continue
                
            unique_vals = np.unique(gt_img)
            # 正常應該是 0-18 以及 255
            # 如果出現 > 18 且不等於 255 的數字，代表轉換腳本沒跑好 (還是 0-33)
            invalid_mask = (unique_vals > 18) & (unique_vals != 255)
            if np.any(invalid_mask):
                print(f"❌ GT 數值異常 (Line {idx}): 發現 {unique_vals[invalid_mask]} (應該要在 0-18 之間)")
                invalid_labels += 1

    # 總結報告
    print("-" * 30)
    print(f"📊 檢查結果 ({csv_path}):")
    if missing_files == 0 and mismatched_names == 0 and invalid_labels == 0:
        print("✅ 通過所有檢查！資料集看起來很健康。")
    else:
        print(f"❌ 遺失檔案數: {missing_files}")
        print(f"⚠️ 檔名不匹配: {mismatched_names}")
        print(f"❌ GT 數值錯誤 (未轉換?): {invalid_labels}")

def visualize_sample(img_path, ref_path, gt_path, idx):
    """將一組圖片畫出來，讓你肉眼確認"""
    plt.figure(figsize=(15, 5))
    
    # 1. 霧天原圖
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Input (Foggy)\nIndex: {idx}")
    plt.axis('off')

    # 2. 參考遮罩 (RGB)
    if pd.notna(ref_path):
        ref = cv2.imread(ref_path)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 2)
        plt.imshow(ref)
        plt.title("Reference (Color)")
        plt.axis('off')

    # 3. GT (TrainID)
    if pd.notna(gt_path):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # 為了視覺化，將 255 轉為黑色(0)，並把 ID 乘上係數讓它亮一點
        vis_gt = gt.copy()
        vis_gt[vis_gt == 255] = 0 
        plt.subplot(1, 3, 3)
        plt.imshow(vis_gt, cmap='nipy_spectral', vmin=0, vmax=19) # 使用彩色映射區分 ID
        plt.title("GT (TrainID 0-18)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    # 如果是在 Server 無法顯示視窗，可以改存檔:
    # plt.savefig(f"verify_sample_{idx}.png")
    # plt.close()

if __name__ == "__main__":
    for csv in CSV_FILES:
        check_csv(csv)