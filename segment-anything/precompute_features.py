# precompute_features.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os
import argparse
import cv2

from segment_anything.build_pair_sam import build_pair_sam_vit_h, build_pair_sam_vit_b


def preprocess_image(image, pixel_mean, pixel_std, img_size):
    image = (image - pixel_mean) / pixel_std
    h, w = image.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    image = F.pad(image, (0, padw, 0, padh))
    return image


def encode_column(df, image_col, output_col, feature_dir, image_encoder,
                  pixel_mean, pixel_std, img_size, device):
    """對 df 中 image_col 欄位的每張圖跑 ViT-H，結果寫入 output_col。"""
    os.makedirs(feature_dir, exist_ok=True)
    feature_paths = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{image_col} → {output_col}"):
            image_path = str(row[image_col])
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Cannot read {image_path}, skipping...")
                feature_paths.append(None)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_size, img_size))
            input_tensor = torch.as_tensor(image, device=device).permute(2, 0, 1).float()
            input_tensor = preprocess_image(input_tensor, pixel_mean, pixel_std, img_size).unsqueeze(0)

            features = image_encoder(input_tensor).squeeze(0).cpu()  # (256, 64, 64)

            filename = os.path.basename(image_path).rsplit('.', 1)[0] + '.pt'
            save_path = os.path.join(feature_dir, filename)
            torch.save(features, save_path)
            feature_paths.append(os.path.abspath(save_path))

    df[output_col] = feature_paths
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file",   type=str,
                        default="/home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_val.csv")
    parser.add_argument("--output_csv", type=str,
                        default="/home/rvl1421/SAM_research-1/Datasets/acdc_val_with_embeddings.csv")
    parser.add_argument("--foggy_feature_dir", type=str,
                        default="/home/rvl1421/SAM_research-1/Datasets/acdc_foggy_features_val")
    parser.add_argument("--clear_feature_dir", type=str,
                        default="/home/rvl1421/SAM_research-1/Datasets/acdc_clear_features_val")
    parser.add_argument("--foggy_image_col",   type=str, default="image_path")
    parser.add_argument("--clear_image_col",   type=str, default="ref_image_path")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type",  type=str, default="vit_h")
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. 載入模型
    print(f"🏗️  Loading {args.model_type} Image Encoder from {args.checkpoint} ...")
    if args.model_type == "vit_h":
        sam_model = build_pair_sam_vit_h(checkpoint=args.checkpoint)
    else:
        sam_model = build_pair_sam_vit_b(checkpoint=args.checkpoint)

    sam_model.to(args.device)
    image_encoder = sam_model.image_encoder
    image_encoder.eval()

    pixel_mean = sam_model.pixel_mean
    pixel_std  = sam_model.pixel_std
    img_size   = image_encoder.img_size

    # 2. 讀取 CSV
    df = pd.read_csv(args.csv_file)
    print(f"📂 Loaded {len(df)} rows from {args.csv_file}")

    for col in [args.foggy_image_col, args.clear_image_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # 3. 惡劣天氣 embedding → feature_path
    print(f"\n[1/2] Encoding foggy images ({args.foggy_image_col}) ...")
    df = encode_column(df, args.foggy_image_col, "feature_path",
                       args.foggy_feature_dir, image_encoder,
                       pixel_mean, pixel_std, img_size, args.device)

    # 4. 晴天 embedding → clear_feature_path
    print(f"\n[2/2] Encoding clear images ({args.clear_image_col}) ...")
    df = encode_column(df, args.clear_image_col, "clear_feature_path",
                       args.clear_feature_dir, image_encoder,
                       pixel_mean, pixel_std, img_size, args.device)

    # 5. 儲存
    df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Done! Output CSV saved to: {args.output_csv}")
    print(f"   • feature_path       → {args.foggy_feature_dir}")
    print(f"   • clear_feature_path → {args.clear_feature_dir}")


if __name__ == "__main__":
    main()
