import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_and_visualize_colors(image_path, top_n=10):
    """
    分析圖片中的顏色分佈，並產出色卡圖表。
    args:
        image_path: 圖片路徑
        top_n: 要列出前幾名最多的顏色 (預設 10)
    """
    if not os.path.exists(image_path):
        print(f"錯誤: 找不到圖片 {image_path}")
        return

    # 1. 讀取圖片 (OpenCV 預設是 BGR)
    img = cv2.imread(image_path)
    # 轉為 RGB (非常重要！因為 SAM 吃的是 RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. 攤平並找出唯一顏色
    reshaped = img.reshape(-1, 3)
    unique_colors, counts = np.unique(reshaped, axis=0, return_counts=True)
    
    # 3. 排序 (面積由大到小)
    sorted_indices = np.argsort(counts)[::-1]
    
    print(f"正在分析圖片: {image_path}")
    print(f"顯示前 {top_n} 種顏色...")
    
    # 4. 準備繪圖
    # 建立一個圖表，高度根據顏色數量動態調整
    fig, ax = plt.subplots(figsize=(8, top_n * 1.2))
    ax.axis('off') # 關閉座標軸
    ax.set_xlim(0, 4)
    ax.set_ylim(0, top_n)
    
    print("------------------------------------------------")
    print("   Count   |   RGB Color      ")
    print("------------------------------------------------")

    # 5. 逐一畫出顏色
    for i in range(min(top_n, len(unique_colors))):
        idx = sorted_indices[i]
        color = unique_colors[idx]
        count = counts[idx]
        
        # 正規化顏色到 0~1 之間 (Matplotlib 使用)
        color_norm = color / 255.0
        
        # (A) 在終端機印出數據
        print(f"{count:9d}  |  {str(color):14s}")
        
        # (B) 在圖表上畫出顏色矩形 (Color Swatch)
        # 參數: (x, y), width, height
        rect = plt.Rectangle((0, top_n - i - 1 + 0.1), 1, 0.8, color=color_norm)
        ax.add_patch(rect)
        
        # (C) 標註文字信息
        text_info = f"RGB: {list(color)}\nCount: {count}"
        
        # 簡單推測類別 (你可以根據觀察結果修改這裡的註解)
        # 這裡只是範例，實際請看圖確認
        hint = ""
        if i == 0: hint = "(Maybe Background/Road?)"
        
        ax.text(1.2, top_n - i - 0.5, f"{text_info} {hint}", fontsize=12, va='center')

    # 6. 存檔
    output_filename = "color_analysis_result.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    print("------------------------------------------------")
    print(f"✅ 分析完成！色卡圖表已儲存為: {output_filename}")
    plt.close()

# --- 執行區 ---
TARGET_IMAGE = "data/weather_dataset/train/masks/02877960.png" 

if __name__ == "__main__":
    analyze_and_visualize_colors(TARGET_IMAGE)