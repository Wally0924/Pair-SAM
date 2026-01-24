import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引用你的模組
from segment_anything.build_weather_sam import build_weather_sam_vit_h, build_weather_sam_vit_b
from utils.weather_dataloader import WeatherSegmentationDataset

# 定義 Cityscapes 顏色表 (用於視覺化)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

class InferenceRunner:
    def __init__(self, model, device, output_dir="inference_results"):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定義類別列表 (順序必須固定)
        self.classes = [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle"
        ]
        self.num_classes = len(self.classes)

    def colorize_mask(self, mask):
        """將類別 Index Mask 轉換為 RGB 圖像"""
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_id in range(self.num_classes):
            color_mask[mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
        return color_mask

    @torch.no_grad()
    def predict_single_image(self, sample):
        """
        對單張影像進行全類別預測
        Return: (H, W) 的 class index map
        """
        # 1. 基礎輸入 (Mask & Original Size)
        # reference_mask 本身通常是 (3, H, W)，不需要 unsqueeze，讓 WeatherSAM 內部處理
        base_input = {
            'reference_mask': sample['reference_mask'].to(self.device),
            'original_size': (1024, 1024) # 推論時統一在 1024 空間操作
        }

        # 2. 動態加入影像輸入
        if 'image_embedding' in sample and sample['image_embedding'] is not None:
            base_input['image_embedding'] = sample['image_embedding'].to(self.device)
        elif 'image' in sample:
            base_input['image'] = sample['image'].to(self.device)
        else:
            raise ValueError("Sample must contain either 'image' or 'image_embedding'")

        # 3. 策略：一次輸入所有 Prompts (Batch Prompting)
        base_input['text_prompts'] = self.classes 
        
        # Forward Pass
        # outputs[0] 對應 batch 中的第一張圖
        outputs = self.model([base_input], multimask_output=True) 
        result = outputs[0]
        
        low_res_logits = result['low_res_logits']   # (19, 3, 256, 256)
        iou_preds = result['iou_predictions']       # (19, 3)
        
        # 針對每個類別 (19個)，選出 3 個 Mask 中 IoU 分數最高的那一個
        best_idx = torch.argmax(iou_preds, dim=1) # (19,)
        
        final_class_logits = []
        for cls_i in range(self.num_classes):
            idx = best_idx[cls_i]
            best_logit = low_res_logits[cls_i, idx, :, :] 
            final_class_logits.append(best_logit)
            
        # 堆疊並上採樣
        final_class_logits = torch.stack(final_class_logits, dim=0) # (19, 256, 256)
        
        full_res_logits = F.interpolate(
            final_class_logits.unsqueeze(0), 
            size=(1024, 1024), 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0) # (19, 1024, 1024)

        # 最終預測: Argmax
        pred_mask = torch.argmax(full_res_logits, dim=0) # (1024, 1024)
        
        # Resize 回原始影像尺寸 (從 sample 中讀取)
        original_size = sample['original_size']
        if original_size != (1024, 1024):
            pred_mask = F.interpolate(
                pred_mask.float().unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='nearest'
            ).long().squeeze()

        return pred_mask.cpu().numpy()

    def visualize(self, sample, pred_mask, idx):
        """
        視覺化：
        Layout:
          [ Input Image ] [ Prediction ]
          [          Legend            ]
        """
        
        # 1. 準備預測的彩色圖
        pred_color = self.colorize_mask(pred_mask)
        
        # 2. 準備輸入影像
        if 'image' in sample:
            img = sample['image'].permute(1, 2, 0).cpu().numpy()
            # [修正] 直接除以 255.0 轉為 0-1
            img = img / 255.0
            img = np.clip(img, 0, 1)
            
            # Resize
            target_h, target_w = pred_mask.shape
            if img.shape[:2] != (target_h, target_w):
                img = cv2.resize(img, (target_w, target_h))
        else:
            img = np.full((*pred_mask.shape, 3), 0.5, dtype=np.float32)

        # 3. 設定畫布與 GridSpec
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(5, 2, figure=fig)
        
        # --- 左上: 原圖 (佔據前 4 列，第 0 行) ---
        ax1 = fig.add_subplot(gs[0:4, 0])
        ax1.imshow(img)
        ax1.set_title("Input Image", fontsize=14)
        ax1.axis('off')

        # --- 右上: 預測結果 (佔據前 4 列，第 1 行) ---
        ax2 = fig.add_subplot(gs[0:4, 1])
        ax2.imshow(pred_color)
        ax2.set_title("Prediction (WeatherSAM)", fontsize=14)
        ax2.axis('off')

        # --- 下方: 圖例 (佔據最後 1 列，跨越所有行) ---
        ax_legend = fig.add_subplot(gs[4, :])
        ax_legend.axis('off')
        
        # 找出這張圖中實際預測出的類別
        unique_classes = np.unique(pred_mask)
        legend_patches = []
        
        for cls_id in unique_classes:
            if cls_id >= self.num_classes: continue
            
            class_name = self.classes[cls_id]
            color = CITYSCAPES_PALETTE[cls_id] / 255.0 
            patch = mpatches.Patch(color=color, label=f"{class_name}")
            legend_patches.append(patch)
        
        if legend_patches:
            ax_legend.legend(
                handles=legend_patches, 
                loc='center', 
                ncol=min(len(legend_patches), 6), # 最多並排 6 個，超過換行
                frameon=False, 
                fontsize='large',
                title="Predicted Classes"
            )
        else:
            ax_legend.text(0.5, 0.5, "No classes detected", ha='center', va='center')

        plt.tight_layout()
        
        # 存檔
        save_path = os.path.join(self.output_dir, f"result_{idx:03d}.png")
        plt.savefig(save_path)
        plt.close()

    def run_inference(self, test_loader, num_samples=None):
        """執行純推論與視覺化 (不計算指標)"""
        self.model.eval()
        
        samples_processed = 0
        pbar = tqdm(test_loader, desc="Inference")
        
        for batch in pbar:
            # 處理 Batch Size = 1 的情況
            sample = {
                'reference_mask': batch['reference_mask'][0],
                'original_size': batch['original_size'][0]
            }
            if 'image' in batch:
                sample['image'] = batch['image'][0]
            if 'image_embedding' in batch:
                sample['image_embedding'] = batch['image_embedding'][0]
                
            # 1. 預測
            pred_mask = self.predict_single_image(sample)
            
            # 2. 視覺化
            self.visualize(sample, pred_mask, idx=samples_processed)
            
            samples_processed += 1
            if num_samples and samples_processed >= num_samples:
                break
        
        print(f"\n✅ Inference completed. Results saved to: {self.output_dir}")

# --- Main Execution ---
if __name__ == "__main__":
    # 設定路徑
    CHECKPOINT_PATH = "/home/rvl1421/SAM_research/segment-anything/outputs_weather_sam/weather_sam_best.pth" # 請確認你的權重檔名
    TEST_CSV_PATH = "/home/rvl1421/SAM_research/Datasets/test.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 載入模型
    print("Loading Model...")
    model = build_weather_sam_vit_h(checkpoint=CHECKPOINT_PATH)
    model.to(DEVICE)
    
    # 2. 準備數據
    test_ds = WeatherSegmentationDataset(csv_file=TEST_CSV_PATH, mode='test')
    test_loader = DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn
    )
    
    # 3. 執行推論
    runner = InferenceRunner(model, DEVICE, output_dir="inference_viz_cityscapes_2layers")
    # 只跑前 100 張，若要跑全部請設 num_samples=None
    runner.run_inference(test_loader, num_samples=50)