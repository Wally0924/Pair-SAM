import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import random
import os

# 引入專案模組
from segment_anything.build_weather_sam import build_weather_sam_vit_h, build_weather_sam_vit_b
from utils.weather_dataloader import WeatherSegmentationDataset

def show_mask(mask, ax, color_code):
    """
    將遮罩疊加在 Matplotlib 的軸上
    color_code: [R, G, B, Alpha] (0~1)
    """
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color_code).reshape(1, 1, 4)
    ax.imshow(mask_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your trained .pth file")
    parser.add_argument("--csv_path", type=str, default="/home/rvl1421/SAM_research/Datasets/val.csv", help="Path to validation CSV")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_h", "vit_b"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index", type=int, default=-1, help="Specific index to check. -1 for random.")
    args = parser.parse_args()

    print(f"🔍 Loading model: {args.model_type}...")
    
    # 1. 建立模型架構 (不載入原始 SAM 權重，因為我們要載入微調後的)
    if args.model_type == "vit_h":
        model = build_weather_sam_vit_h(checkpoint=None)
    else:
        model = build_weather_sam_vit_b(checkpoint=None)
    
    # 2. 載入訓練好的權重
    print(f"📂 Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # 處理可能的 key 前綴問題 (如 module. 或 image_encoder.)
    # 這裡假設存檔是直接存 model.state_dict()
    model.load_state_dict(checkpoint, strict=False) 
    model.to(args.device)
    model.eval()

    # 3. 準備資料集
    # 注意：這裡我們強制不使用 cache 特徵讀取原圖，以便視覺化
    # 如果你的 CSV 只有 cache 路徑沒有 image_path，這裡可能會報錯
    # 建議使用包含 image_path 的原始 CSV，或確保 Dataset class 能處理
    real_csv_path = args.csv_path
    if "_cached" in real_csv_path and not os.path.exists(real_csv_path):
        real_csv_path = real_csv_path.replace("_cached.csv", ".csv")

    ds = WeatherSegmentationDataset(csv_file=real_csv_path, mode='val')
    
    # 選擇樣本
    if args.index == -1:
        idx = random.randint(0, len(ds)-1)
    else:
        idx = args.index
    
    print(f"🖼️  Inspecting Sample Index: {idx}")
    sample = ds[idx]

    # 4. 建構輸入 (模擬 Collate Function)
    # 因為 Dataset 可能回傳 image_embedding (若用 cache csv)，我們需要處理兩種情況
    batched_input = []
    input_dict = {
        'reference_mask': sample['reference_mask'].to(args.device), # (1, 3, H, W)
        'text_prompts': sample['text_prompts'],
        'original_size': sample['original_size']
    }

    # 處理影像來源
    if 'image' in sample:
        input_dict['image'] = sample['image'].to(args.device) # (3, H, W)
        # 用於顯示的原圖
        vis_image = sample['image'].permute(1, 2, 0).numpy()
        # Denormalize for visualization (粗略還原)
        vis_image = vis_image / 255.0
        vis_image = np.clip(vis_image, 0, 1)
    elif 'image_embedding' in sample:
        input_dict['image_embedding'] = sample['image_embedding'].to(args.device)
        # 如果是 Embedding 模式，我們沒辦法顯示原圖，就顯示全黑或 Reference Mask 代替
        vis_image = np.zeros((1024, 1024, 3))
        print("⚠️ Warning: Using cached features. Original image not available for visualization.")

    batched_input.append(input_dict)

    # 5. 推論 (Inference)
    with torch.no_grad():
        # multimask_output=False 讓我們只看模型最有信心的那個結果
        outputs = model(batched_input, multimask_output=False)

    # 6. 解析輸出
    # outputs[0]['masks'] shape: (K, 1, H, W) -> K 是 Prompt 數量
    # outputs[0]['iou_predictions'] shape: (K, 1)
    pred_masks = outputs[0]['masks']
    iou_preds = outputs[0]['iou_predictions']
    prompts = sample['text_prompts']
    gt_mask = sample['gt_mask'].numpy()

    # 7. 繪圖
    num_prompts = len(prompts)
    fig, axes = plt.subplots(num_prompts, 4, figsize=(20, 5 * num_prompts))
    if num_prompts == 1: axes = axes.reshape(1, -1)

    for i, prompt in enumerate(prompts):
        ax_row = axes[i]
        
        # A. 原圖
        ax_row[0].imshow(vis_image)
        ax_row[0].set_title(f"Input Image\nPrompt: '{prompt}'")
        ax_row[0].axis('off')

        # B. Reference Mask (記憶)
        ref_vis = sample['reference_mask'].permute(1, 2, 0).numpy()
        ref_vis = ref_vis / 255.0  # [修正 2] 將 0-255 轉為 0-1 以避免 Clipping 警告
        ax_row[1].imshow(ref_vis)
        ax_row[1].set_title("Reference Mask (Memory)")
        ax_row[1].axis('off')

        # C. Ground Truth (真實答案)
        # 找出對應這個 prompt 的 class ID
        target_id = ds.CLASS_MAP.get(prompt, -1)
        gt_binary = (gt_mask == target_id).astype(float)
        
        ax_row[2].imshow(vis_image)
        show_mask(gt_binary, ax_row[2], [0, 1, 0, 0.6]) # 綠色是 GT
        ax_row[2].set_title(f"Ground Truth ({prompt})")
        ax_row[2].axis('off')

        # D. Model Prediction (預測)
        # Sigmoid -> Threshold
        pred_binary = pred_masks[i, 0, :, :].cpu().numpy().astype(float)
        
        conf_score = iou_preds[i, 0].item()

        ax_row[3].imshow(vis_image)
        show_mask(pred_binary, ax_row[3], [1, 0, 0, 0.6]) # 紅色是預測
        ax_row[3].set_title(f"Prediction (IoU Conf: {conf_score:.2f})")
        ax_row[3].axis('off')

    plt.tight_layout()
    output_file = f"inference_check_{idx}.png"
    plt.savefig(output_file)
    print(f"✅ Result saved to {output_file}")

if __name__ == "__main__":
    main()