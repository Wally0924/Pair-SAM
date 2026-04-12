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
        
        # 全域 mIoU 累積器（Cityscapes 標準算法）
        self.total_intersection = np.zeros(self.num_classes)
        self.total_union = np.zeros(self.num_classes)

    def colorize_mask(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_id in range(self.num_classes):
            color_mask[mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
        return color_mask

    def calculate_miou(self, pred_mask, gt_mask, ignore_index=255):
        """計算單張圖片的 mIoU（用於即時顯示），同時累積全域統計量。"""
        ious = []
        valid_mask = gt_mask != ignore_index
        for cls_id in range(self.num_classes):
            pred_cls = (pred_mask == cls_id)
            gt_cls = (gt_mask == cls_id)
            intersection = np.logical_and(pred_cls, gt_cls)[valid_mask].sum()
            union = np.logical_or(pred_cls, gt_cls)[valid_mask].sum()
            
            # 全域累積（Cityscapes 標準）
            self.total_intersection[cls_id] += intersection
            self.total_union[cls_id] += union
            
            if union > 0:
                ious.append(intersection / union)
        return np.mean(ious) if len(ious) > 0 else 0.0

    def compute_global_miou(self):
        """計算全資料集的 mIoU（Cityscapes 標準：全域累積後再取類別平均）。"""
        per_class_iou = {}
        valid_ious = []
        for cls_id in range(self.num_classes):
            if self.total_union[cls_id] > 0:
                iou = self.total_intersection[cls_id] / self.total_union[cls_id]
                per_class_iou[self.classes[cls_id]] = iou
                valid_ious.append(iou)
            else:
                per_class_iou[self.classes[cls_id]] = float('nan')
        
        global_miou = np.mean(valid_ious) if valid_ious else 0.0
        return global_miou, per_class_iou

    @torch.no_grad()
    def predict_single_image(self, sample, active_prompts):
        """透過自定義 Predictor 進行推論（與訓練一致的 Soft Selection）"""
        
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
        
        # 2. 進行預測 — 取得 low_res_masks (K, 3, 256, 256) 和 iou_preds (K, 3)
        masks, iou_preds, low_res_masks = self.predictor.predict(
            text_prompts=active_prompts,
            location=sample['location'].to(self.device),
            multimask_output=True
        )
        
        # 3. Argmax 選取最佳 candidate（與訓練一致）
        best_mask_indices = torch.argmax(iou_preds, dim=1)  # (K,)
        selected_logits = low_res_masks[torch.arange(len(active_prompts)), best_mask_indices]  # (K, 256, 256)
        
        # 4. list + cat 組裝 19 class channels at 256²
        class_channels = []
        for cls_id in range(self.num_classes):
            matched = False
            for k, prompt in enumerate(active_prompts):
                if self.classes.index(prompt) == cls_id:
                    class_channels.append(selected_logits[k].unsqueeze(0))
                    matched = True
                    break
            if not matched:
                class_channels.append(torch.full((1, 256, 256), -10.0, device=self.device))
        
        full_class_logits = torch.cat(class_channels, dim=0).unsqueeze(0)  # (1, 19, 256, 256)
        
        # 5. FusionHead at 256²
        fused_logits = self.predictor.model.context_fusion_head(full_class_logits)
        
        # 6. postprocess_masks 上採樣到 original_size（正確處理 padding）
        orig_h, orig_w = sample['original_size']
        fused_logits = self.predictor.model.postprocess_masks(
            fused_logits,
            input_size=(1024, 1024),
            original_size=(orig_h, orig_w),
        )
        
        # 7. Argmax 決策
        fused_logits = fused_logits.squeeze(0)  # (19, H, W)
        pred_mask = torch.argmax(fused_logits, dim=0)  # (H, W)
        
        # 濾波：如果所有 class logit 都極低，標記為 ignore (255)
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
        all_mious = []
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
                all_mious.append(miou)
                print(f"📊 Image {samples_processed:03d} | mIoU: {miou:.4f}")
            
            self.visualize(sample, pred_mask, gt_resized_np, idx=samples_processed, miou=miou)
            
            samples_processed += 1
            if num_samples is not None and samples_processed >= num_samples:
                break
        
        # 計算並輸出全域 mIoU（Cityscapes 標準算法）
        if samples_processed > 0:
            global_miou, per_class_iou = self.compute_global_miou()
            print(f"\n{'='*60}")
            print(f"🏆 Global mIoU (Cityscapes Standard): {global_miou:.4f}  ({samples_processed} images)")
            print(f"{'='*60}")
            print(f"{'Class':<20} {'IoU':>10}")
            print(f"{'-'*30}")
            for cls_name, iou_val in per_class_iou.items():
                if np.isnan(iou_val):
                    print(f"{cls_name:<20} {'N/A':>10}")
                else:
                    print(f"{cls_name:<20} {iou_val:>10.4f}")
            print(f"{'-'*30}")
            print(f"{'mIoU':<20} {global_miou:>10.4f}")
            print(f"{'='*60}")

if __name__ == "__main__":
    CHECKPOINT_PATH = "/home/rvl1421/SAM_research-1/segment-anything/outputs_weather_sam_all_data_testv16/weather_sam_best_latest.pth"
    TEST_CSV_PATH = "/home/rvl1421/SAM_research-1/Datasets/test_with_gps.csv" 
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
    
    runner = InferenceRunner(predictor, DEVICE, output_dir="inference_viz_cityscapes_testv16")
    runner.run_inference(test_loader, num_samples=None)  # num_samples=None → 跑完整個 test dataset