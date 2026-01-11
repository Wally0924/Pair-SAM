import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from tabulate import tabulate  # 需要 pip install tabulate
import pandas as pd

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.build_sam import build_sam_vit_h

# ==========================================
# 1. 設定區域 (CONFIG)
# ==========================================
CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "IMG_SIZE": 1024,
    "NUM_TEST_SAMPLES": 300,  # 要測試幾張圖 (設為 None 則測試全部)
    
    # 資料路徑
    "TEST_IMG_DIR": "data/weather_dataset/train/images",
    "TEST_MASK_DIR": "data/weather_dataset/train/masks",
    
    # 模型路徑
    # A. 官方原始權重 (Baseline)
    "ORIGINAL_CHECKPOINT": "checkpoints/sam_vit_h_4b8939.pth", 
    # B. 你訓練好的權重 (Fine-tuned)
    "FINETUNED_CHECKPOINT": "outputs/sam_weather_best.pth",
}

# ==========================================
# 2. 評估指標模組 (Metrics Module)
# ==========================================
class SegmentationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ious = []
        self.dices = []
        self.precisions = []
        self.recalls = []

    def update(self, pred_mask, gt_mask):
        """
        計算單張影像的指標
        pred_mask: bool or 0/1 numpy array
        gt_mask: bool or 0/1 numpy array
        """
        # 確保輸入是 boolean
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()

        # 1. IoU (Intersection over Union)
        iou = intersection / (union + 1e-6)
        
        # 2. Dice Coefficient (F1 Score)
        dice = (2. * intersection) / (pred_sum + gt_sum + 1e-6)
        
        # 3. Precision (預測為正的樣本中，有多少是真的正)
        precision = intersection / (pred_sum + 1e-6)
        
        # 4. Recall (真的正樣本中，有多少被預測出來)
        recall = intersection / (gt_sum + 1e-6)

        self.ious.append(iou)
        self.dices.append(dice)
        self.precisions.append(precision)
        self.recalls.append(recall)

    def get_averages(self):
        return {
            "mIoU": np.mean(self.ious) * 100,
            "Dice": np.mean(self.dices) * 100,
            "Precision": np.mean(self.precisions) * 100,
            "Recall": np.mean(self.recalls) * 100
        }

# ==========================================
# 3. 輔助函式 (Utils)
# ==========================================
def get_box_from_mask(mask):
    """從 Mask 計算 Bounding Box"""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 加入一點隨機擾動 (Perturbation) 模擬真實情況，也讓測試更嚴苛
        # 如果你不想要擾動，把 noise 設為 0
        noise = 0 
        h, w = mask.shape
        x_min = max(0, x_min - noise)
        x_max = min(w, x_max + noise)
        y_min = max(0, y_min - noise)
        y_max = min(h, y_max + noise)
        
        return np.array([x_min, y_min, x_max, y_max])
    return None

def load_model(checkpoint_path, model_type="vit_h", device="cuda"):
    """載入模型 (通用函式)"""
    print(f"正在載入模型: {checkpoint_path} ...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到權重檔: {checkpoint_path}")
        
    # 建立骨架
    sam = build_sam_vit_h(checkpoint=None)
    
    # 載入權重
    state_dict = torch.load(checkpoint_path, map_location=device)
    sam.load_state_dict(state_dict)
    sam.to(device)
    sam.eval() # 設定為評估模式
    return SamPredictor(sam)

# ==========================================
# 4. 核心評估迴圈 (Evaluation Loop)
# ==========================================
def evaluate_model(predictor, image_files, config, name="Model"):
    metrics = SegmentationMetrics()
    
    print(f"\n🚀 開始評估: {name}")
    pbar = tqdm(image_files, desc=f"Evaluating {name}")
    
    for img_name in pbar:
        # 路徑設定
        img_path = os.path.join(config["TEST_IMG_DIR"], img_name)
        mask_path = os.path.join(config["TEST_MASK_DIR"], os.path.splitext(img_name)[0] + ".png")
        
        if not os.path.exists(mask_path):
            continue

        # 1. 讀取與前處理
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        
        # 強制縮放 (保持與訓練一致)
        image = cv2.resize(image, (config["IMG_SIZE"], config["IMG_SIZE"]))
        mask = cv2.resize(mask, (config["IMG_SIZE"], config["IMG_SIZE"]), interpolation=cv2.INTER_NEAREST)

        # 2. 處理 RGB Mask -> 找出主要物件
        # 這裡簡化邏輯：找出圖中最大的那個非黑色的物件顏色
        reshaped_mask = mask.reshape(-1, 3)
        unique_colors, counts = np.unique(reshaped_mask, axis=0, return_counts=True)
        valid_indices = [i for i, c in enumerate(unique_colors) if not np.array_equal(c, [0, 0, 0])]
        
        if len(valid_indices) == 0: continue
            
        largest_idx = valid_indices[np.argmax(counts[valid_indices])]
        target_color = unique_colors[largest_idx]
        gt_binary_mask = np.all(mask == target_color, axis=2)

        # 3. 產生 Prompt Box (基於 GT)
        input_box = get_box_from_mask(gt_binary_mask)
        if input_box is None: continue

        # 4. 模型推論
        predictor.set_image(image)
        pred_masks, scores, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=False
        )
        pred_binary_mask = pred_masks[0]

        # 5. 更新指標
        metrics.update(pred_binary_mask, gt_binary_mask)

    return metrics.get_averages()

# ==========================================
# 5. 主程式 (Main)
# ==========================================
def main():
    # 準備測試資料列表
    all_files = sorted([f for f in os.listdir(CONFIG["TEST_IMG_DIR"]) if f.endswith(('.jpg', '.png'))])
    if CONFIG["NUM_TEST_SAMPLES"]:
        import random
        random.seed(42)
        random.shuffle(all_files)
        test_files = all_files[:CONFIG["NUM_TEST_SAMPLES"]]
    else:
        test_files = all_files

    print(f"測試集數量: {len(test_files)} 張")

    # --- 評估 1: 原始模型 (Vanilla SAM) ---
    predictor_orig = load_model(CONFIG["ORIGINAL_CHECKPOINT"], device=CONFIG["DEVICE"])
    results_orig = evaluate_model(predictor_orig, test_files, CONFIG, name="Original SAM")
    
    # 釋放顯存
    del predictor_orig
    torch.cuda.empty_cache()

    # --- 評估 2: 微調模型 (Fine-tuned SAM) ---
    predictor_fine = load_model(CONFIG["FINETUNED_CHECKPOINT"], device=CONFIG["DEVICE"])
    results_fine = evaluate_model(predictor_fine, test_files, CONFIG, name="Fine-tuned SAM")

    # --- 產生比較表格 ---
    print("\n\n" + "="*50)
    print("📊 實驗結果比較 (Test Set)")
    print("="*50)

    # 整理資料
    df = pd.DataFrame([results_orig, results_fine], index=["Original SAM (Baseline)", "Fine-tuned SAM (Ours)"])
    
    # 計算進步幅度
    improvement = df.loc["Fine-tuned SAM (Ours)"] - df.loc["Original SAM (Baseline)"]
    df.loc["Difference"] = improvement

    # 顯示表格
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
    
    print("\n[指標說明]")
    print("* mIoU (Mean Intersection over Union): 分割準確度的黃金標準，越高越好。")
    print("* Dice (F1 Score): 綜合考量 Precision 與 Recall，對不平衡資料較敏感。")
    print("* Precision: 預測出的 Mask 中，有多少是真的物體 (防止切太大)。")
    print("* Recall: 真實物體中，有多少被 Mask 覆蓋到 (防止切太小)。")

if __name__ == "__main__":
    main()