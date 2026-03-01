

# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.gridspec as gridspec
# from tqdm import tqdm
# from torch.utils.data import DataLoader

# # 引用你的模組
# from segment_anything.build_weather_sam import build_weather_sam_vit_h, build_weather_sam_vit_b
# from utils.weather_dataloader import WeatherSegmentationDataset

# # 定義 Cityscapes 顏色表
# CITYSCAPES_PALETTE = np.array([
#     [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
#     [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
#     [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
#     [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
#     [0, 80, 100], [0, 0, 230], [119, 11, 32]
# ], dtype=np.uint8)

# class InferenceRunner:
#     def __init__(self, model, device, output_dir="inference_results"):
#         self.model = model
#         self.device = device
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
        
#         self.classes = [
#             "road", "sidewalk", "building", "wall", "fence",
#             "pole", "traffic light", "traffic sign", "vegetation",
#             "terrain", "sky", "person", "rider", "car",
#             "truck", "bus", "train", "motorcycle", "bicycle"
#         ]
#         self.num_classes = len(self.classes)

#     def colorize_mask(self, mask):
#         """將類別 Index Mask 轉換為 RGB 圖像"""
#         color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#         for cls_id in range(self.num_classes):
#             color_mask[mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
#         return color_mask

#     # ==========================================
#     # 單圖 mIoU 計算邏輯
#     # ==========================================
#     def calculate_miou(self, pred_mask, gt_mask, ignore_index=255):
#         """計算單張影像的 mIoU，忽略 ignore_index"""
#         ious = []
#         valid_mask = gt_mask != ignore_index
        
#         for cls_id in range(self.num_classes):
#             pred_cls = (pred_mask == cls_id)
#             gt_cls = (gt_mask == cls_id)
            
#             # 計算交集與聯集 (僅在有效區域內)
#             intersection = np.logical_and(pred_cls, gt_cls)[valid_mask].sum()
#             union = np.logical_or(pred_cls, gt_cls)[valid_mask].sum()
            
#             if union > 0:
#                 ious.append(intersection / union)
                
#         if len(ious) == 0:
#             return 0.0
#         return np.mean(ious)

#     @torch.no_grad()
#     def predict_single_image(self, sample, active_prompts):
#         # 1. 基礎輸入 (維持不變)
#         base_input = {
#             'reference_mask': sample['reference_mask'].to(self.device),
#             'ref_void_mask': sample['ref_void_mask'].to(self.device),
#             'location': sample['location'].to(self.device),
#             'original_size': (1024, 1024)
#         }

#         if 'image_embedding' in sample and sample['image_embedding'] is not None:
#             base_input['image_embedding'] = sample['image_embedding'].to(self.device)
#         elif 'image' in sample:
#             base_input['image'] = sample['image'].to(self.device)
#         else:
#             raise ValueError("Sample must contain either 'image' or 'image_embedding'")

#         # 🌟 2. [關鍵修改] 只傳入這張圖實際存在的類別 (模擬訓練環境)
#         base_input['text_prompts'] = active_prompts 
        
#         # 3. 執行推論
#         outputs = self.model([base_input], multimask_output=True) 
#         result = outputs[0]
        
#         # 取得模型預測結果
#         low_res_logits = result['low_res_logits'] # (K, 3, 256, 256)，K 是 active_prompts 的數量
#         iou_preds = result['iou_predictions']     # (K, 3)
        
#         best_idx = torch.argmax(iou_preds, dim=1) # (K,)
        
#         # 🌟 4. [關鍵修改] 建立一個乾淨的 19 類底布，預設分數為極小值 (-1000)
#         # 代表「預設所有類別都不存在」
#         final_class_logits = torch.full((self.num_classes, 256, 256), -1000.0, device=self.device)
        
#         # 🌟 5. [關鍵修改] 將有預測的 K 個類別的 Logit，精準填入它在 19 類中的對應位置
#         for k, prompt in enumerate(active_prompts):
#             cls_i = self.classes.index(prompt)
#             idx = best_idx[k]
#             final_class_logits[cls_i] = low_res_logits[k, idx, :, :]
            
#         # 6. 上採樣與 Argmax
#         full_res_logits = F.interpolate(
#             final_class_logits.unsqueeze(0), 
#             size=(1024, 1024), 
#             mode="bilinear", 
#             align_corners=False
#         ).squeeze(0)

#         max_logits, pred_mask = torch.max(full_res_logits, dim=0)

#         pred_mask[max_logits < 0.0] = 255
        
#         # 7. Resize 回原尺寸
#         original_size = sample['original_size']
#         if original_size != (1024, 1024):
#             pred_mask = F.interpolate(
#                 pred_mask.float().unsqueeze(0).unsqueeze(0),
#                 size=original_size,
#                 mode='nearest'
#             ).long().squeeze()

#         return pred_mask.cpu().numpy()

#     def visualize(self, sample, pred_mask, gt_np, idx, miou=None):
#         """視覺化 2x2 Grid"""
#         if 'image' in sample:
#             img = sample['image'].permute(1, 2, 0).cpu().numpy()
#             img = img / 255.0
#             img = np.clip(img, 0, 1)
#         else:
#             img = np.full((*pred_mask.shape, 3), 0.5, dtype=np.float32)

#         ref_img = sample['reference_mask'].permute(1, 2, 0).cpu().numpy()
#         ref_img = ref_img / 255.0
#         ref_img = np.clip(ref_img, 0, 1)

#         pred_color = self.colorize_mask(pred_mask)
        
#         target_h, target_w = pred_mask.shape
#         if gt_np is not None:
#             gt_color = self.colorize_mask(gt_np)
#         else:
#             gt_color = np.zeros_like(pred_color)

#         if img.shape[:2] != (target_h, target_w):
#             img = cv2.resize(img, (target_w, target_h))
#         if ref_img.shape[:2] != (target_h, target_w):
#             ref_img = cv2.resize(ref_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

#         fig = plt.figure(figsize=(16, 12)) 
        
#         if miou is not None:
#             fig.suptitle(f"Sample {idx:03d} | Image mIoU: {miou:.4f}", fontsize=20, fontweight='bold', y=0.98)

#         gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.15], figure=fig)
        
#         ax1 = fig.add_subplot(gs[0, 0])
#         ax1.imshow(img)
#         ax1.set_title("Input Image (Foggy)", fontsize=14)
#         ax1.axis('off')

#         ax2 = fig.add_subplot(gs[0, 1])
#         ax2.imshow(ref_img)
#         ax2.set_title("Reference Mask (Memory)", fontsize=14)
#         ax2.axis('off')

#         ax3 = fig.add_subplot(gs[1, 0])
#         ax3.imshow(pred_color)
#         ax3.set_title("Prediction (WeatherSAM)", fontsize=14)
#         ax3.axis('off')

#         ax4 = fig.add_subplot(gs[1, 1])
#         ax4.imshow(gt_color)
#         ax4.set_title("Ground Truth", fontsize=14)
#         ax4.axis('off')

#         ax_legend = fig.add_subplot(gs[2, :])
#         ax_legend.axis('off')
        
#         unique_classes = set(np.unique(pred_mask))
#         if gt_np is not None:
#             unique_classes.update(np.unique(gt_np))
            
#         legend_patches = []
#         for cls_id in sorted(list(unique_classes)):
#             if cls_id >= self.num_classes: continue
#             class_name = self.classes[cls_id]
#             color = CITYSCAPES_PALETTE[cls_id] / 255.0 
#             patch = mpatches.Patch(color=color, label=f"{class_name}")
#             legend_patches.append(patch)
        
#         if legend_patches:
#             ax_legend.legend(
#                 handles=legend_patches, 
#                 loc='center', 
#                 ncol=min(len(legend_patches), 8), 
#                 frameon=False, 
#                 fontsize='medium',
#                 title="Classes Present"
#             )

#         plt.tight_layout()
#         if miou is not None:
#             plt.subplots_adjust(top=0.92)
            
#         save_path = os.path.join(self.output_dir, f"result_{idx:03d}.png")
#         plt.savefig(save_path)
#         plt.close()

#     def run_inference(self, test_loader, num_samples=None):
#         self.model.eval()
#         samples_processed = 0
#         pbar = tqdm(test_loader, desc="Inference")
        
#         for batch in pbar:
#             sample = {
#                 'reference_mask': batch['reference_mask'][0],
#                 'ref_void_mask': batch['ref_void_mask'][0],
#                 'location': batch['location'][0],
#                 'original_size': batch['original_size'][0]
#             }
#             if 'image' in batch:
#                 sample['image'] = batch['image'][0]
#             if 'image_embedding' in batch:
#                 sample['image_embedding'] = batch['image_embedding'][0]
            
#             # 🌟 [關鍵修改] 動態萃取 Active Prompts
#             gt_mask_for_prompt = batch['gt_mask'][0].numpy()
#             active_prompts = []
            
#             # 從 GT 中找出存在的類別
#             if gt_mask_for_prompt.max() > 0:
#                 unique_classes = np.unique(gt_mask_for_prompt)
#                 for cls_id in unique_classes:
#                     if cls_id < self.num_classes: # 過濾掉 255
#                         active_prompts.append(self.classes[cls_id])
            
#             # 防呆：如果完全沒東西，給個預設值
#             if not active_prompts:
#                 active_prompts = ["road"]

#             # 🌟 傳入 active_prompts 給模型
#             pred_mask = self.predict_single_image(sample, active_prompts)
            
#             # ... (下方的計算 mIoU 與視覺化程式碼完全保持原樣不變) ...
#             gt_resized_np = None
#             miou = None
            
#             if 'gt_mask' in batch:
#                 gt_mask = batch['gt_mask'][0].to(self.device)
#                 target_h, target_w = pred_mask.shape
#                 gt_tensor = gt_mask.unsqueeze(0).unsqueeze(0).float()
#                 gt_resized_np = F.interpolate(
#                     gt_tensor, size=(target_h, target_w), mode='nearest'
#                 ).long().squeeze().cpu().numpy()
                
#                 miou = self.calculate_miou(pred_mask, gt_resized_np)
#                 print(f"📊 Image {samples_processed:03d} | mIoU: {miou:.4f}")
            
#             self.visualize(sample, pred_mask, gt_resized_np, idx=samples_processed, miou=miou)
            
#             samples_processed += 1
#             if num_samples and samples_processed >= num_samples:
#                 break
        
#         print(f"\n✅ Inference completed. Results saved to: {self.output_dir}")

# # --- Main Execution ---
# if __name__ == "__main__":
#     CHECKPOINT_PATH = "/home/rvl1421/SAM_research/segment-anything/outputs_weather_sam_all_data_testv7/best_E28_Dice0.7136_LR9.7e-06.pth"
#     TEST_CSV_PATH = "/home/rvl1421/SAM_research/Datasets/test_with_gps.csv" 
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     print("Loading Model...")
#     model = build_weather_sam_vit_h(checkpoint=CHECKPOINT_PATH)
#     model.to(DEVICE)
    
#     test_ds = WeatherSegmentationDataset(csv_file=TEST_CSV_PATH, mode='val') 
#     test_ds.has_cached_features = False
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=1, 
#         shuffle=False, 
#         num_workers=4,
#         collate_fn=WeatherSegmentationDataset.collate_fn
#     )
    
#     runner = InferenceRunner(model, DEVICE, output_dir="inference_viz_cityscapes_testv7")
#     runner.run_inference(test_loader, num_samples=10)




import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from torch.utils.data import DataLoader

from segment_anything.build_weather_sam import build_weather_sam_vit_h, build_weather_sam_vit_b
from utils.weather_dataloader import WeatherSegmentationDataset
from segment_anything.weather_predictor import WeatherSamPredictor  # 👈 引入剛剛建立的 Predictor

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

class InferenceRunner:
    def __init__(self, predictor: WeatherSamPredictor, device, output_dir="inference_results"):
        self.predictor = predictor
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.classes = [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle"
        ]
        self.num_classes = len(self.classes)

    def colorize_mask(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_id in range(self.num_classes):
            color_mask[mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
        return color_mask

    def calculate_miou(self, pred_mask, gt_mask, ignore_index=255):
        ious = []
        valid_mask = gt_mask != ignore_index
        for cls_id in range(self.num_classes):
            pred_cls = (pred_mask == cls_id)
            gt_cls = (gt_mask == cls_id)
            intersection = np.logical_and(pred_cls, gt_cls)[valid_mask].sum()
            union = np.logical_or(pred_cls, gt_cls)[valid_mask].sum()
            if union > 0:
                ious.append(intersection / union)
        return np.mean(ious) if len(ious) > 0 else 0.0

    @torch.no_grad()
    def predict_single_image(self, sample, active_prompts):
        """透過自定義 Predictor 進行推論"""
        
        image_tensor = sample.get('image', None)
        image_embedding = sample.get('image_embedding', None)
        if image_tensor is not None: image_tensor = image_tensor.to(self.device)
        if image_embedding is not None: image_embedding = image_embedding.to(self.device)

        # 1. 設定影像資料 (計算並快取特徵)
        self.predictor.set_image_data(
            image=image_tensor,
            image_embedding=image_embedding,
            reference_mask=sample['reference_mask'].to(self.device),
            ref_void_mask=sample['ref_void_mask'].to(self.device),
            original_size=sample['original_size']
        )
        
        # 2. 進行預測
        # masks 已經被 Predictor 上採樣並對齊回 original_size，維度為 (K, 3, H, W)
        masks, iou_preds, _ = self.predictor.predict(
            text_prompts=active_prompts,
            location=sample['location'].to(self.device),
            multimask_output=True
        )
        
        best_idx = torch.argmax(iou_preds, dim=1) # 找出每個類別預測最佳的 Mask 索引
        
        # 3. 建立畫布 (大小為 Original Size)
        orig_h, orig_w = sample['original_size']
        final_class_logits = torch.full((self.num_classes, orig_h, orig_w), -1000.0, device=self.device)
        
        # 4. 精準填入預測 Logits
        for k, prompt in enumerate(active_prompts):
            cls_i = self.classes.index(prompt)
            idx = best_idx[k]
            # 這裡的 masks 已經透過 postprocess_masks 處理過且對齊 original_size
            final_class_logits[cls_i] = masks[k, idx, :, :] 
            
        # 5. Semantic Fusion Head 互斥預測
        # 將 (19, H, W) 擴充為 (1, 19, H, W) 送入 Head
        fused_logits = self.predictor.model.semantic_fusion_head(final_class_logits.unsqueeze(0))
        
        # 6. 決策邊界 (Argmax 直接得到互斥結果)
        fused_logits = fused_logits.squeeze(0) # (19, H, W)
        pred_mask = torch.argmax(fused_logits, dim=0) # (H, W)
        
        # 濾波機制 (如果最大機率依然小於某個閾值，判定為背景 255)
        # 這裡我們用 Softmax 或直接看 logit 值。如果是 CrossEntropy 訓練的，logit < 0 可能不適用。
        # 但為了保險起見，如果所有 class 的 logit 都極低 (例如都小於 0)，我們還是將其設為 255。
        max_logits, _ = torch.max(fused_logits, dim=0)
        pred_mask[max_logits < 0.0] = 255 

        return pred_mask.cpu().numpy()

    def visualize(self, sample, pred_mask, gt_np, idx, miou=None):
        """視覺化 2x2 Grid"""
        if 'image' in sample:
            img = sample['image'].permute(1, 2, 0).cpu().numpy()
            img = img / 255.0
            img = np.clip(img, 0, 1)
        else:
            img = np.full((*pred_mask.shape, 3), 0.5, dtype=np.float32)

        ref_img = sample['reference_mask'].permute(1, 2, 0).cpu().numpy()
        ref_img = ref_img / 255.0
        ref_img = np.clip(ref_img, 0, 1)

        pred_color = self.colorize_mask(pred_mask)
        
        target_h, target_w = pred_mask.shape
        if gt_np is not None:
            gt_color = self.colorize_mask(gt_np)
        else:
            gt_color = np.zeros_like(pred_color)

        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h))
        if ref_img.shape[:2] != (target_h, target_w):
            ref_img = cv2.resize(ref_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        fig = plt.figure(figsize=(16, 12)) 
        
        if miou is not None:
            fig.suptitle(f"Sample {idx:03d} | Image mIoU: {miou:.4f}", fontsize=20, fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.15], figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title("Input Image (Foggy)", fontsize=14)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(ref_img)
        ax2.set_title("Reference Mask (Memory)", fontsize=14)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(pred_color)
        ax3.set_title("Prediction (WeatherSAM)", fontsize=14)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(gt_color)
        ax4.set_title("Ground Truth", fontsize=14)
        ax4.axis('off')

        ax_legend = fig.add_subplot(gs[2, :])
        ax_legend.axis('off')
        
        unique_classes = set(np.unique(pred_mask))
        if gt_np is not None:
            unique_classes.update(np.unique(gt_np))
            
        legend_patches = []
        for cls_id in sorted(list(unique_classes)):
            if cls_id >= self.num_classes: continue
            class_name = self.classes[cls_id]
            color = CITYSCAPES_PALETTE[cls_id] / 255.0 
            patch = mpatches.Patch(color=color, label=f"{class_name}")
            legend_patches.append(patch)
        
        if legend_patches:
            ax_legend.legend(
                handles=legend_patches, 
                loc='center', 
                ncol=min(len(legend_patches), 8), 
                frameon=False, 
                fontsize='medium',
                title="Classes Present"
            )

        plt.tight_layout()
        if miou is not None:
            plt.subplots_adjust(top=0.92)
            
        save_path = os.path.join(self.output_dir, f"result_{idx:03d}.png")
        plt.savefig(save_path)
        plt.close()

    def run_inference(self, test_loader, num_samples=None):
        samples_processed = 0
        pbar = tqdm(test_loader, desc="Inference")
        
        for batch in pbar:
            sample = {
                'reference_mask': batch['reference_mask'][0],
                'ref_void_mask': batch['ref_void_mask'][0],
                'location': batch['location'][0],
                'original_size': batch['original_size'][0]
            }
            if 'image' in batch:
                sample['image'] = batch['image'][0]
            if 'image_embedding' in batch:
                sample['image_embedding'] = batch['image_embedding'][0]
            
            gt_mask_for_prompt = batch['gt_mask'][0].numpy()
            active_prompts = []
            if gt_mask_for_prompt.max() > 0:
                unique_classes = np.unique(gt_mask_for_prompt)
                for cls_id in unique_classes:
                    if cls_id < self.num_classes:
                        active_prompts.append(self.classes[cls_id])
            
            if not active_prompts:
                active_prompts = ["road"]

            # 呼叫改寫後的推論函式
            pred_mask = self.predict_single_image(sample, active_prompts)
            
            gt_resized_np = None
            miou = None
            if 'gt_mask' in batch:
                gt_mask = batch['gt_mask'][0].to(self.device)
                target_h, target_w = pred_mask.shape
                gt_tensor = gt_mask.unsqueeze(0).unsqueeze(0).float()
                import torch.nn.functional as F
                gt_resized_np = F.interpolate(
                    gt_tensor, size=(target_h, target_w), mode='nearest'
                ).long().squeeze().cpu().numpy()
                
                miou = self.calculate_miou(pred_mask, gt_resized_np)
                print(f"📊 Image {samples_processed:03d} | mIoU: {miou:.4f}")
            
            self.visualize(sample, pred_mask, gt_resized_np, idx=samples_processed, miou=miou)
            
            samples_processed += 1
            if num_samples and samples_processed >= num_samples:
                break

if __name__ == "__main__":
    CHECKPOINT_PATH = "/home/rvl1421/SAM_research-1/segment-anything/outputs_weather_sam_all_data_testv7/weather_sam_best_latest.pth"
    TEST_CSV_PATH = "/home/rvl1421/SAM_research/Datasets/test_with_gps.csv" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Model...")
    model = build_weather_sam_vit_h(checkpoint=CHECKPOINT_PATH)
    model.to(DEVICE)
    
    # 實例化新的 Predictor
    predictor = WeatherSamPredictor(model)
    
    test_ds = WeatherSegmentationDataset(csv_file=TEST_CSV_PATH, mode='val') 
    test_ds.has_cached_features = False
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn
    )
    
    runner = InferenceRunner(predictor, DEVICE, output_dir="inference_viz_cityscapes_testv7")
    runner.run_inference(test_loader, num_samples=10)