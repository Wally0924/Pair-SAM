import os
import pandas as pd
from tqdm import tqdm

# ================= 設定路徑 (請修改這裡) =================
# train + val 有 3475 張影像，test 有 1525 張影像
# 1. 霧天影像根目錄
FOGGY_ROOT = "/home/rvl1421/Datasets/Cityscapes_foggy/leftImg8bit_foggy"

# 2. GT/Ref 根目錄
GT_ROOT = "/home/rvl1421/Datasets/Cityscapes/GT/gtFine"

# 3. 輸出位置 (存到你的專案目錄)
OUTPUT_DIR = "/home/rvl1421/SAM_research/Datasets"

# 4. 霧濃度後綴列表 (擴增：包含 0.02, 0.01, 0.005)
# 程式會掃描這些後綴，將符合的影像都加入資料集
FOGGY_SUFFIXES = [
    "_foggy_beta_0.02.png", 
    "_foggy_beta_0.01.png", 
    "_foggy_beta_0.005.png"
]
# =======================================================

def process_split(split_name):
    """處理單一 Split 並回傳 DataFrame"""
    print(f"🔄 正在處理: {split_name} ...")
    
    foggy_dir = os.path.join(FOGGY_ROOT, split_name)
    gt_dir = os.path.join(GT_ROOT, split_name)
    
    if not os.path.exists(foggy_dir):
        print(f"⚠️  警告: 找不到目錄 {foggy_dir}，跳過此 split。")
        return None

    data_list = []
    
    # 遍歷該 split 下的所有城市 (加入 sorted 確保順序固定)
    cities = sorted(os.listdir(foggy_dir))
    
    for city in tqdm(cities, desc=f"Scanning {split_name}"):
        city_foggy_path = os.path.join(foggy_dir, city)
        city_gt_path = os.path.join(gt_dir, city)
        
        if not os.path.isdir(city_foggy_path): continue

        # 遍歷影像 (加入 sorted 確保順序固定，不打亂)
        files = sorted(os.listdir(city_foggy_path))
        
        for f in files:
            # 檢查檔案是否符合我們定義的任一種霧濃度
            matched_suffix = None
            for suffix in FOGGY_SUFFIXES:
                if f.endswith(suffix):
                    matched_suffix = suffix
                    break
            
            if matched_suffix:
                # 1. 影像路徑 (不同濃度的霧圖)
                img_path = os.path.join(city_foggy_path, f)
                
                # 2. 解析核心 ID (Core ID) 以連結到共用的 GT
                # 原始檔名結構: frankfurt_000000_000294_leftImg8bit_foggy_beta_0.02.png
                # 去除後綴後得到 Core ID: frankfurt_000000_000294
                core_id = f.replace("_leftImg8bit" + matched_suffix, "")
                
                # 3. 建構對應路徑 (共用 GT)
                ref_name = f"{core_id}_gtFine_color.png"
                label_name = f"{core_id}_gtFine_labelTrainIds.png"
                
                # 若是 test 集，gt_dir 可能不存在或沒有檔案
                ref_path = os.path.join(city_gt_path, ref_name) if os.path.exists(city_gt_path) else None
                label_path = os.path.join(city_gt_path, label_name) if os.path.exists(city_gt_path) else None
                
                # 4. 建立資料 Entry
                entry = {
                    "image_path": img_path,
                    "ref_mask_path": "",
                    "gt_path": ""
                }
                
                # 檢查 Reference Mask
                if ref_path and os.path.exists(ref_path):
                    entry["ref_mask_path"] = ref_path
                
                # 檢查 Label (GT)
                if label_path and os.path.exists(label_path):
                    entry["gt_path"] = label_path
                
                # 邏輯判斷：
                # Train/Val: 必須有 GT 才能訓練，嚴格檢查
                # Test: 通常沒有 GT，允許空白 (只供推論用)
                if split_name in ['train', 'val']:
                    if entry["ref_mask_path"] and entry["gt_path"]:
                        data_list.append(entry)
                else:
                    # Test 集：只要有原圖就收錄
                    data_list.append(entry)

    return pd.DataFrame(data_list)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 分別生成三個檔案，嚴格按照原始資料集的劃分
    for split in ['train', 'val', 'test']:
        df = process_split(split)
        
        if df is not None and not df.empty:
            save_path = os.path.join(OUTPUT_DIR, f"{split}.csv")
            # index=False, 不進行 shuffle，保留原始遍歷順序
            df.to_csv(save_path, index=False)
            print(f"✅ 已儲存: {save_path} (共 {len(df)} 筆)")
        else:
            print(f"⚠️  {split} 沒有找到有效資料或目錄不存在。")

if __name__ == "__main__":
    main()