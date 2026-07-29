import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.dataloader import PairSegmentationDataset

# ================= 設定區 =================
CONFIG = {
    "DATA_ROOT": "data/weather_dataset/train", # 請確認路徑正確
    "MAX_SAMPLES": 5,  # 檢查幾張圖
    "SAVE_PATH": "debug_point_prompt.png"
}
# =========================================

def denormalize_image(image_tensor):
    """將 Tensor 反標準化回 0-255 的圖片"""
    image = image_tensor.permute(1, 2, 0).numpy()
    pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
    image = (image * pixel_std + pixel_mean)
    return np.clip(image, 0, 255).astype(np.uint8)

def show_points(coords, labels, ax, marker_size=375):
    """畫出提示點"""
    # coords shape: (N, 2)
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    
    # 畫出綠色星號代表前景點
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # 畫出紅色點代表背景點 (如果有)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def main():
    if not os.path.exists(CONFIG["DATA_ROOT"]):
        print(f"❌ 錯誤：找不到資料夾 {CONFIG['DATA_ROOT']}")
        return

    # 1. 建立 Dataset
    print("正在初始化 Dataset (這可能需要幾秒鐘)...")
    dataset = PairSegmentationDataset(root_dir=CONFIG["DATA_ROOT"], mode='train')
    
    # 2. 隨機取樣
    indices = torch.randperm(len(dataset))[:CONFIG["MAX_SAMPLES"]]
    
    # 設定畫布大小
    plt.figure(figsize=(18, 6 * CONFIG["MAX_SAMPLES"]))
    
    print(f"🚀 開始檢查 {CONFIG['MAX_SAMPLES']} 筆樣本...")
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # 還原數據
        image = denormalize_image(sample['image'])
        mask = sample['mask'].squeeze().numpy() # (1024, 1024)
        
        # 讀取點座標與標籤
        point_coords = sample['point_coords'].numpy() # (1, 2)
        point_labels = sample['point_labels'].numpy() # (1,)
        
        # 判斷這張圖大概是在訓練什麼
        mask_area = np.sum(mask > 0)
        if mask_area == 0:
            label_guess = "Empty (Background)"
        elif mask_area < 5000: # 面積很小，通常是車道線
            label_guess = "Likely: Lane Marking"
        else: # 面積很大，通常是路面或人行道
            label_guess = "Likely: Road / Sidewalk"

        # --- 繪圖 (三欄位) ---
        
        # Col 1: Input Image + Point Prompt
        ax1 = plt.subplot(CONFIG["MAX_SAMPLES"], 3, i*3 + 1)
        ax1.imshow(image)
        # 畫點
        show_points(point_coords, point_labels, ax1)
        ax1.set_title(f"Sample {i}: Point Prompt\n({label_guess})", fontsize=14)
        ax1.axis('off')
        
        # Col 2: Ground Truth Mask (黑底白字)
        ax2 = plt.subplot(CONFIG["MAX_SAMPLES"], 3, i*3 + 2)
        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Target Mask (Model Output)", fontsize=14)
        ax2.axis('off')
        
        # Col 3: Overlay (疊合檢查對齊)
        ax3 = plt.subplot(CONFIG["MAX_SAMPLES"], 3, i*3 + 3)
        ax3.imshow(image)
        
        # 製作半透明紅色遮罩
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask * 255 # 紅色通道
        
        # 只在有 Mask 的地方疊色
        ax3.imshow(colored_mask, alpha=0.6) 
        # 疊上點
        show_points(point_coords, point_labels, ax3)
        
        ax3.set_title("Overlay Check", fontsize=14)
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig(CONFIG["SAVE_PATH"])
    print(f"\n✅ 檢查完成！圖片已儲存至: {CONFIG['SAVE_PATH']}")
    print("------------------------------------------------------")
    print("🔍 檢查重點：")
    print("1. [Point] 綠色星號是否準確落在白色 Mask 區域內？")
    print("2. [Lane] 車道線是否只是一小段，且點就在那一小段上？")
    print("3. [Box] (已移除) 現在我們只看點！")
    print("------------------------------------------------------")

if __name__ == "__main__":
    main()