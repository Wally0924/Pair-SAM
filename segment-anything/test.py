import numpy as np
import torch
import cv2
import os
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def main():
    # 1. 設定參數
    CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
    MODEL_TYPE = "vit_h"
    IMAGE_PATH = "notebooks/images/groceries.jpg"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 設定結果輸出資料夾
    OUTPUT_DIR = "results_all"
    MASKS_DIR = os.path.join(OUTPUT_DIR, "individual_masks")
    os.makedirs(MASKS_DIR, exist_ok=True)

    print(f"🚀 正在使用裝置: {DEVICE}")

    # 2. 載入 SAM 模型
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未檢測到 CUDA，速度會非常慢！")
    
    print("⏳ 正在載入模型...")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        
        # === 關鍵修改：改用自動遮罩產生器 ===
        # 這裡可以調整 points_per_side 來控制檢測的細緻度，預設是 32
        mask_generator = SamAutomaticMaskGenerator(sam)
        
        print("✅ 模型載入成功！")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    # 3. 讀取圖片
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        print(f"❌ 找不到圖片: {IMAGE_PATH}")
        return
    
    # 轉為 RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 4. 全自動分割 (這步會比較久，取決於圖片複雜度和顯卡)
    print("🤖 正在進行全圖自動分割 (Segment Everything)...")
    masks = mask_generator.generate(image_rgb)
    
    print(f"✨ 分割完成！總共偵測到 {len(masks)} 個物件。")

    # 5. 視覺化處理
    print("🎨 正在繪製結果...")

    # 準備一張底圖用來畫彩色遮罩總覽
    combined_img = image_bgr.copy()
    
    # 依序處理每一個遮罩
    for i, mask_data in enumerate(masks):
        # mask_data 是一個字典，其中 'segmentation' 鍵存放的是布林遮罩
        bool_mask = mask_data['segmentation']
        
        # --- A. 儲存獨立的黑白遮罩 ---
        # 為了避免檔案太多，我們只存面積大於一定程度的，或全部存
        single_mask_filename = os.path.join(MASKS_DIR, f"mask_{i:03d}.png")
        cv2.imwrite(single_mask_filename, (bool_mask * 255).astype(np.uint8))

        # --- B. 製作彩色總覽圖 ---
        # 產生一個隨機顏色 (BGR)
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        
        # 在原圖上疊加顏色
        # 建立一個與圖片同大小的純色圖層
        colored_mask = np.zeros_like(combined_img, dtype=np.uint8)
        colored_mask[:] = color
        
        # 只在遮罩區域應用顏色
        alpha = 0.5 # 透明度
        
        # 這裡使用 numpy 的遮罩索引功能進行快速混合
        # 公式： 原圖區域 = 原圖區域 * (1-alpha) + 顏色 * alpha
        # 注意：需要先轉成 float 運算再轉回 uint8
        
        region = combined_img[bool_mask]
        combined_img[bool_mask] = (region * (1 - alpha) + color * alpha).astype(np.uint8)

    # 6. 儲存總覽圖
    overlay_path = os.path.join(OUTPUT_DIR, "combined_result.jpg")
    cv2.imwrite(overlay_path, combined_img)

    print(f"💾 總覽圖已儲存: {overlay_path}")
    print(f"💾 {len(masks)} 張獨立遮罩已儲存至: {MASKS_DIR}")

if __name__ == "__main__":
    main()