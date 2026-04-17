# precompute_features.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import argparse
import cv2
import numpy as np

# 引用現有模組
from segment_anything.build_weather_sam import build_weather_sam_vit_h, build_weather_sam_vit_b
from utils.weather_dataloader import WeatherSegmentationDataset

#要跑過 train 跟　val 都提取特徵。

def preprocess_image(image, pixel_mean, pixel_std, img_size):
    """手動執行 WeatherSAM 的預處理邏輯 (Normalization + Padding)"""
    # Normalize
    image = (image - pixel_mean) / pixel_std
    
    # Pad
    h, w = image.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    image = F.pad(image, (0, padw, 0, padh))
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="/home/rvl1421/SAM_research/Datasets/val_all.csv", help="Path to input CSV (e.g., data/weather_dataset/train.csv)")
    parser.add_argument("--output_csv", type=str, default="/home/rvl1421/SAM_research/Datasets/val_all_cached.csv", help="Path to save new CSV with feature paths")
    parser.add_argument("--feature_dir", type=str, default="/home/rvl1421/SAM_research/Datasets/features_val_all", help="Directory to save .pt files")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # [image-pair] 指定要預算的影像欄位與輸出欄位名稱
    # 預設行為（foggy）：--image_col image_path --output_col feature_path
    # clear-weather：  --image_col clear_image_path --output_col clear_feature_path
    parser.add_argument("--image_col", type=str, default="image_path",
                        help="CSV column name for input image paths")
    parser.add_argument("--output_col", type=str, default="feature_path",
                        help="CSV column name to write output feature paths")
    args = parser.parse_args()

    # 1. 建立目錄
    os.makedirs(args.feature_dir, exist_ok=True)
    
    # 2. 載入模型 (只取 Image Encoder)
    print(f"🏗️  Loading {args.model_type} Image Encoder...")
    if args.model_type == "vit_h":
        sam_model = build_weather_sam_vit_h(checkpoint=args.checkpoint)
    else:
        sam_model = build_weather_sam_vit_b(checkpoint=args.checkpoint)
        
    sam_model.to(args.device)
    image_encoder = sam_model.image_encoder
    image_encoder.eval()

    # 獲取預處理參數 (Mean/Std)
    pixel_mean = sam_model.pixel_mean
    pixel_std = sam_model.pixel_std
    img_size = image_encoder.img_size

    # 3. 讀取 CSV
    df = pd.read_csv(args.csv_file)
    if args.image_col not in df.columns:
        raise ValueError(f"Column '{args.image_col}' not found in CSV. Available: {list(df.columns)}")
    print(f"📂 Processing {len(df)} images from '{args.image_col}' column in {args.csv_file}")

    feature_paths = []

    # 4. 開始轉換
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_path = row[args.image_col]
            
            # 讀取圖片
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Error reading {image_path}, skipping...")
                feature_paths.append(None)
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize 到 1024 (保持與訓練時一致)
            image = cv2.resize(image, (img_size, img_size))
            
            # 轉 Tensor: (H, W, 3) -> (3, H, W)
            input_image = torch.as_tensor(image, device=args.device).permute(2, 0, 1).float()
            
            # Preprocess (Normalize + Pad)
            # 注意：這裡輸入要有 Batch 維度 -> (1, 3, 1024, 1024)
            input_image = preprocess_image(input_image, pixel_mean, pixel_std, img_size).unsqueeze(0)
            
            # Forward Pass (ViT)
            # Output Shape: (1, 256, 64, 64)
            features = image_encoder(input_image)
            
            # 移除 Batch 維度以節省空間 -> (256, 64, 64)
            features = features.squeeze(0).cpu()
            
            # 存檔
            # 檔名範例: features/train_0001.pt
            filename = os.path.basename(image_path).replace('.jpg', '.pt').replace('.png', '.pt')
            save_path = os.path.join(args.feature_dir, filename)
            torch.save(features, save_path)
            
            # 記錄絕對路徑
            feature_paths.append(os.path.abspath(save_path))

    # 5. 更新 CSV
    df[args.output_col] = feature_paths
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Done! Column '{args.output_col}' written. New CSV saved to: {args.output_csv}")

if __name__ == "__main__":
    main()