import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from utils.dataloader import WeatherSegmentationDataset

# ================= 設定區 =================
# 請改成你的資料路徑
DATA_ROOT = "data/weather_dataset/train" 
MAX_SAMPLES = 5  # 檢查前 5 張就好
# =========================================

def show_box(box, ax):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def main():
    # 1. 建立 Dataset (模擬訓練模式)
    dataset = WeatherSegmentationDataset(root_dir=DATA_ROOT, mode='train')
    
    # 2. 隨機取樣
    indices = torch.randperm(len(dataset))[:MAX_SAMPLES]
    
    plt.figure(figsize=(15, 5 * MAX_SAMPLES))
    
    print(f"正在檢查 {MAX_SAMPLES} 筆訓練樣本...")
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # 還原 Image (Tensor -> Numpy)
        image = sample['image'].permute(1, 2, 0).numpy()
        # 反標準化 (De-normalization) 以便視覺化
        pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
        pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
        image = (image * pixel_std + pixel_mean).astype(np.uint8)
        
        # Mask (Tensor -> Numpy)
        mask = sample['mask'].squeeze().numpy()
        
        # Box
        box = sample['box'].numpy()
        
        # --- 繪圖 ---
        # 1. 影像 + Box
        ax1 = plt.subplot(MAX_SAMPLES, 3, i*3 + 1)
        ax1.imshow(image)
        show_box(box, ax1)
        ax1.set_title(f"Sample {i}: Input Image + Prompt Box")
        ax1.axis('off')
        
        # 2. Ground Truth Mask (模型被教導要輸出的樣子)
        ax2 = plt.subplot(MAX_SAMPLES, 3, i*3 + 2)
        ax2.imshow(mask, cmap='gray')
        ax2.set_title(f"GT Mask (Target)\nMax Val: {mask.max()}")
        ax2.axis('off')
        
        # 3. 疊合顯示
        ax3 = plt.subplot(MAX_SAMPLES, 3, i*3 + 3)
        ax3.imshow(image)
        # 畫上半透明紅色 Mask
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask * 255 # 紅色通道
        ax3.imshow(colored_mask, alpha=0.5)
        show_box(box, ax3)
        ax3.set_title("Overlay Check")
        ax3.axis('off')

    save_path = "debug_dataloader_sample.png"
    plt.savefig(save_path)
    print(f"✅ 檢查完成！請打開圖片 '{save_path}' 確認：")
    print("1. GT Mask 是否只是一堆雜訊點？ (如果是 -> RGB 雜訊問題)")
    print("2. GT Mask 是否全黑？ (如果是 -> 空遮罩問題)")
    print("3. Box 是否包住物體？")

if __name__ == "__main__":
    main()