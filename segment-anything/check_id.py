import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 假設你的 Dataset 類別名稱為 WeatherDataset，請根據你的 pair_dataloader.py 實際名稱修改
from utils.pair_dataloader import PairSegmentationDataset 

def verify_class_mapping():
    print("🔍 啟動 Ground Truth 標籤驗證程式...\n")
    
    # 1. 初始化你的 Dataset (請確認你的 CSV 檔案位置正確)
    # 根據你的 pair_dataloader.py，參數應該是 csv_file, image_size, mode
    dataset = PairSegmentationDataset(
        csv_file="/home/rvl1421/SAM_research-1/Datasets/train_with_gps.csv", # 請改成實際的 CSV 路徑
        image_size=1024,
        mode='train'
    ) 
    
    # 2. 印出 Dataset 中註冊的字典，確認你的設定值
    print("=== 📖 Dataset 內部註冊的 ID_TO_NAME 對照表 ===")
    for cls_id, cls_name in dataset.ID_TO_NAME.items():
        print(f"ID: {cls_id:2d} -> {cls_name}")
    print("==============================================\n")

    # 3. 隨機抽取一張影像來驗證
    sample = dataset[0] # 取第一張圖
    gt_mask = sample['gt_mask'].numpy() # 提取 GT Mask (通常是 2D array: H x W)
    
    # 4. 找出這張 GT 裡面實際存在哪些像素值
    unique_ids = np.unique(gt_mask)
    
    print("=== 🎯 實際從 GT Mask 取出的 Unique 像素值 ===")
    for uid in unique_ids:
        if uid == 255:
            print(f"像素值 {uid}: ignore_index (忽略不計)")
        elif uid in dataset.ID_TO_NAME:
            print(f"像素值 {uid:2d}: {dataset.ID_TO_NAME[uid]}")
        else:
            print(f"⚠️ 警告！發現未知的像素值: {uid}")
            
    # 5. (視覺化) 畫出來看最準！
    plt.figure(figsize=(10, 5))
    
    # 顯示 GT Mask，使用 nipy_spectral colormap 讓不同類別對比強烈
    plt.imshow(gt_mask, cmap='nipy_spectral', vmin=0, vmax=18)
    plt.title("Ground Truth Mask (Colors represent Class IDs)")
    plt.axis('off')
    
    # 加上圖例 (Legend)
    import matplotlib.patches as mpatches
    cmap = plt.cm.nipy_spectral
    patches = []
    for uid in unique_ids:
        if uid != 255 and uid in dataset.ID_TO_NAME:
            # 根據 vmin=0, vmax=18 計算顏色
            color = cmap(uid / 18.0) 
            patches.append(mpatches.Patch(color=color, label=f"{uid}: {dataset.ID_TO_NAME[uid]}"))
            
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("gt_verify_check.png")
    print("\n✅ 驗證影像已儲存為 gt_verify_check.png，請打開確認顏色與物件是否對應正確！")

if __name__ == "__main__":
    verify_class_mapping()