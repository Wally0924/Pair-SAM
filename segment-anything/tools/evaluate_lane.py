import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# 強制使用非互動式後端
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 1. 設定區域 (CONFIG)
# ==========================================
CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "CHECKPOINT": "outputs/sam_weather_best.pth",
    "MODEL_TYPE": "vit_h",
    "TEST_IMG_DIR": "data/weather_dataset/val/images",
    "TEST_MASK_DIR": "data/weather_dataset/val/masks",
    "OUTPUT_DIR": "evaluation_results",
    "NUM_SAMPLES": 10,
    "TARGET_CLASS_ID": [128, 64, 128], # 路面 RGB
}

def calculate_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    return iou, dice

def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    print(f"Loading model...")
    sam = sam_model_registry[CONFIG["MODEL_TYPE"]](checkpoint=CONFIG["CHECKPOINT"])
    sam.to(device=CONFIG["DEVICE"])
    predictor = SamPredictor(sam)

    img_files = sorted(os.listdir(CONFIG["TEST_IMG_DIR"]))[:CONFIG["NUM_SAMPLES"]]
    
    for img_name in tqdm(img_files, desc="Generating Visuals"):
        img_path = os.path.join(CONFIG["TEST_IMG_DIR"], img_name)
        mask_path = os.path.join(CONFIG["TEST_MASK_DIR"], img_name.replace(".jpg", ".png"))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask_rgb = cv2.imread(mask_path)
        gt_mask_rgb = cv2.cvtColor(gt_mask_rgb, cv2.COLOR_BGR2RGB)
        gt_binary = np.all(gt_mask_rgb == CONFIG["TARGET_CLASS_ID"], axis=-1).astype(np.float32)

        if np.sum(gt_binary) < 100: continue

        # 採樣提示點
        y_indices, x_indices = np.where(gt_binary > 0)
        idx = np.random.choice(len(y_indices), 3, replace=False)
        input_points = np.stack([x_indices[idx], y_indices[idx]], axis=1)
        input_labels = np.ones(3)

        # 推理
        predictor.set_image(image)
        masks, _, _ = predictor.predict(input_points, input_labels, multimask_output=False)
        pred_mask = masks[0]
        iou, dice = calculate_metrics(pred_mask, gt_binary)

        # ==========================================
        # 修改重點：調整圖表大小與白邊
        # ==========================================
        # 增加 figsize 的高度比例，並設定子圖間距 wspace 為接近 0
        fig, axes = plt.subplots(1, 4, figsize=(20, 6), gridspec_kw={'wspace': 0.05, 'hspace': 0})
        
        # 1. 原圖 + 提示點
        axes[0].imshow(image)
        axes[0].scatter(input_points[:, 0], input_points[:, 1], color='lime', marker='*', s=100, edgecolors='black')
        axes[0].set_title("Input & Prompts", fontsize=14, pad=10)
        
        # 2. 真值
        axes[1].imshow(gt_binary, cmap='gray')
        axes[1].set_title("Ground Truth", fontsize=14, pad=10)
        
        # 3. 預測
        axes[2].imshow(pred_mask, cmap='viridis')
        axes[2].set_title(f"Pred (IoU: {iou:.3f})", fontsize=14, pad=10)
        
        # 4. 誤差分析
        error_map = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        error_map[(pred_mask == 1) & (gt_binary == 1)] = [0, 255, 0] # TP
        error_map[(pred_mask == 0) & (gt_binary == 1)] = [255, 0, 0] # FN
        error_map[(pred_mask == 1) & (gt_binary == 0)] = [0, 0, 255] # FP
        axes[3].imshow(error_map)
        axes[3].set_title("Error Map", fontsize=14, pad=10)

        # 統一關閉座標軸並去除多餘邊界
        for ax in axes:
            ax.axis('off')

        # 儲存設定：bbox_inches='tight' 是去除白邊的核心
        plt.savefig(
            os.path.join(CONFIG["OUTPUT_DIR"], f"eval_{img_name}"),
            bbox_inches='tight', 
            pad_inches=0.1,
            dpi=150
        )
        plt.close(fig)

    print(f"Results saved to {CONFIG['OUTPUT_DIR']}")

if __name__ == "__main__":
    main()