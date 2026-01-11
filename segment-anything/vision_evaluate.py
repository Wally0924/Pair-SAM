import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.build_sam import build_sam_vit_h

# ==========================================
# 1. 設定區域 (CONFIG)
# ==========================================
CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "IMG_SIZE": 1024,
    "NUM_SAMPLES": 5,  # 想要產生幾張比較圖
    
    # 資料路徑 (請確認路徑正確)
    "TEST_IMG_DIR": "data/weather_dataset/train/images",
    "TEST_MASK_DIR": "data/weather_dataset/train/masks",
    
    # 模型權重路徑
    "ORIGINAL_CHECKPOINT": "checkpoints/sam_vit_h_4b8939.pth", 
    "FINETUNED_CHECKPOINT": "outputs/sam_weather_best.pth",
    
    "OUTPUT_DIR": "final_visual_comparison",
    
    # AMG 參數
    "AMG_KWARGS": {
        "points_per_side": 32,
        "pred_iou_thresh": 0.86,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 0,
        "min_mask_region_area": 100,
    }
}

# ==========================================
# 2. 工具函式
# ==========================================
def show_anns(anns, ax):
    """繪製 SAM 生成的 Masks (拼圖效果)"""
    if len(anns) == 0:
        return
    
    # 依照面積排序
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    h, w = sorted_anns[0]['segmentation'].shape
    img = np.zeros((h, w, 4)) # RGBA
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.6]]) # Alpha = 0.6
        img[m] = color_mask
        
    ax.imshow(img)

def run_amg_inference(checkpoint_path, image_data_list, config):
    """批次推論"""
    print(f"\n[Model] Loading: {checkpoint_path} ...")
    
    sam = build_sam_vit_h(checkpoint=None)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=config["DEVICE"])
        sam.load_state_dict(state_dict)
    sam.to(config["DEVICE"])
    
    mask_generator = SamAutomaticMaskGenerator(model=sam, **config["AMG_KWARGS"])
    
    results = []
    print(f"Running inference on {len(image_data_list)} images...")
    
    for item in image_data_list:
        masks = mask_generator.generate(item['image']) # item['image'] 已經是 RGB 且 Resize 過了
        results.append(masks)
        
    # 釋放顯存
    del sam
    del mask_generator
    torch.cuda.empty_cache()
    
    return results

# ==========================================
# 3. 主程式
# ==========================================
def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    # 1. 準備資料 (同時讀取 Image 和 Mask)
    all_files = sorted([f for f in os.listdir(CONFIG["TEST_IMG_DIR"]) if f.endswith('.jpg')])
    
    # 隨機挑選
    import random
    random.seed(42)
    random.shuffle(all_files)
    target_files = all_files[:CONFIG["NUM_SAMPLES"]]
    
    data_list = []
    print("Loading data...")
    
    for img_name in target_files:
        img_path = os.path.join(CONFIG["TEST_IMG_DIR"], img_name)
        mask_path = os.path.join(CONFIG["TEST_MASK_DIR"], os.path.splitext(img_name)[0] + ".png")
        
        # 讀圖
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
        
        # 讀 Mask (GT)
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path)
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
            # Mask 縮放必須用 INTER_NEAREST 保持邊緣銳利
            gt_mask = cv2.resize(gt_mask, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), interpolation=cv2.INTER_NEAREST)
        else:
            print(f"Warning: GT mask not found for {img_name}")
            gt_mask = np.zeros_like(image) # 給個全黑圖避免報錯

        data_list.append({
            "name": img_name,
            "image": image,
            "gt_mask": gt_mask
        })

    # 2. 執行推論
    # A. Baseline
    results_orig = run_amg_inference(CONFIG["ORIGINAL_CHECKPOINT"], data_list, CONFIG)
    
    # B. Fine-tuned (Ours)
    results_fine = run_amg_inference(CONFIG["FINETUNED_CHECKPOINT"], data_list, CONFIG)

    # 3. 繪圖 (4欄比較圖)
    print(f"\nGenerating comparison charts in {CONFIG['OUTPUT_DIR']} ...")
    
    for i in range(len(data_list)):
        item = data_list[i]
        masks_o = results_orig[i]
        masks_f = results_fine[i]
        
        # 建立畫布：1列 4欄
        fig, axes = plt.subplots(1, 4, figsize=(32, 9))
        
        # Col 1: Original Image
        axes[0].imshow(item['image'])
        axes[0].set_title(f"Input Image\n{item['name']}", fontsize=18)
        
        # Col 2: Ground Truth (直接顯示 RGB Mask)
        axes[1].imshow(item['gt_mask'])
        axes[1].set_title("Ground Truth (Label)", fontsize=18, fontweight='bold')
        
        # Col 3: Baseline
        axes[2].imshow(item['image'])
        show_anns(masks_o, axes[2])
        axes[2].set_title(f"Baseline SAM\n(Count: {len(masks_o)})", fontsize=18, color='darkred')
        
        # Col 4: Ours (Fine-tuned)
        axes[3].imshow(item['image'])
        show_anns(masks_f, axes[3])
        axes[3].set_title(f"Ours (Fine-tuned)\n(Count: {len(masks_f)})", fontsize=18, color='darkgreen', fontweight='bold')
        
        # 關閉座標軸
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        save_name = os.path.join(CONFIG["OUTPUT_DIR"], f"chart_{item['name']}")
        plt.savefig(save_name)
        plt.close()
        print(f"Saved chart: {save_name}")

    print("Done!")

if __name__ == "__main__":
    main()