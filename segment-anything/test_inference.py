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
# [2026-04] 移除 WeatherSamPredictor 依賴：舊 predictor 仍呼叫已不存在的 mask_encoder
# 且 API 未同步 condition_encoder / clear_embedding。改為直接呼叫 model.forward，
# 與 weather_trainer.py 的 eval pipeline 完全對齊。

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

class InferenceRunner:
    def __init__(self, model, device, output_dir="inference_results"):
        # model: WeatherSAM 本體（build_weather_sam_vit_h 回傳物件）
        self.model = model
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
        """
        與 weather_trainer.py eval pipeline 對齊的推論流程：
          1) 組 batched_input（符合 WeatherSAM.forward 介面）→ 取得 low_res_logits (1, K, 256, 256) 與 class_ids
          2) 依 class_ids scatter 至 (1, 19, 256, 256)，未出現的 class 以 -10 填充
          3) context_fusion_head 於 256² 執行 Mask2Former-style 後置精修
          4) postprocess_masks 上採樣至 original_size
          5) argmax 取得 (H, W) 預測；不再套 max_logits<0 過濾（與訓練一致）
        """
        # --- 1. 組 batched_input（單張；欄位名稱需與 WeatherSAM.forward 相符）---
        input_record = {
            "text_prompts": active_prompts,
            "original_size": tuple(sample['original_size']),
            "clear_embedding": sample['clear_embedding'].to(self.device),   # (256, 64, 64)
            "condition_id": sample['condition_id'].to(self.device),         # scalar LongTensor
        }
        if 'image_embedding' in sample and sample['image_embedding'] is not None:
            input_record["image_embedding"] = sample['image_embedding'].to(self.device)
        elif 'image' in sample and sample['image'] is not None:
            input_record["image"] = sample['image'].to(self.device)
        else:
            raise ValueError("Sample 需提供 'image' 或 'image_embedding'。")

        # --- 2. 呼叫 WeatherSAM.forward ---
        outputs = self.model([input_record])
        out = outputs[0]
        low_res_logits = out["low_res_logits"]  # (1, K, 256, 256)
        class_ids = out["class_ids"]            # List[int], len=K；與 active_prompts 順序一致（經過 CLASS_MAP 過濾）

        # --- 3. Scatter 至 (1, 19, 256, 256)；未被 prompt 的 class 以極小值填入 ---
        if low_res_logits.shape[1] == 0:
            # 無有效 prompt：所有 class 皆填 -10，最終 argmax 會落在 class 0；維持與 trainer 一致的防呆
            full_class_logits = torch.full(
                (1, self.num_classes, 256, 256), -10.0, device=self.device
            )
        else:
            selected_logits = low_res_logits[0]  # (K, 256, 256)
            class_channels = []
            cls_id_to_k = {cid: k for k, cid in enumerate(class_ids)}
            for cls_id in range(self.num_classes):
                if cls_id in cls_id_to_k:
                    class_channels.append(selected_logits[cls_id_to_k[cls_id]].unsqueeze(0))
                else:
                    class_channels.append(torch.full((1, 256, 256), -10.0, device=self.device))
            full_class_logits = torch.cat(class_channels, dim=0).unsqueeze(0)  # (1, 19, 256, 256)

        # --- 4. ContextFusionHead @ 256² ---
        fused_logits = self.model.context_fusion_head(full_class_logits)

        # --- 5. 上採樣至 original_size（正確處理 padding）---
        orig_h, orig_w = sample['original_size']
        fused_logits = self.model.postprocess_masks(
            fused_logits,
            input_size=(1024, 1024),
            original_size=(orig_h, orig_w),
        )

        # --- 6. Argmax 決策（不再套 max_logits<0 過濾；trainer 亦未套用）---
        fused_logits = fused_logits.squeeze(0)  # (19, H, W)
        pred_mask = torch.argmax(fused_logits, dim=0)  # (H, W)
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
            # dataloader 的 collate_fn 將 text_prompts 保留為 List[List[str]]、
            # original_size 為 List[Tuple]；其餘為 batched tensor。
            sample = {
                'reference_mask': batch['reference_mask'][0],      # 留作 visualize 使用
                'ref_void_mask': batch['ref_void_mask'][0],
                'location': batch['location'][0],
                'original_size': batch['original_size'][0],
                'clear_embedding': batch['clear_embedding'][0],    # [image-pair] f_ref 來源
                'condition_id': batch['condition_id'][0],          # ACDC fog/rain/snow；Cityscapes = -1
                'invalid_mask': batch['invalid_mask'][0],          # ACDC 固定盲區遮罩（True=無效）
            }
            if 'image' in batch:
                sample['image'] = batch['image'][0]
            if 'image_embedding' in batch:
                sample['image_embedding'] = batch['image_embedding'][0]

            # active_prompts：優先採用 dataloader 產出的 text_prompts（已做 GT→class name 推導）
            # 若為空則 fallback 到 "road"。保留以 GT 決定 prompt 的 oracle 設定，與訓練/驗證一致。
            active_prompts = batch['text_prompts'][0] if batch.get('text_prompts') else []
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

                # 與 trainer 一致：將 invalid_mask 標記的區域設為 255（ignore）
                inv = sample['invalid_mask']  # bool Tensor (H_orig, W_orig)
                if inv.any():
                    inv_np = F.interpolate(
                        inv.unsqueeze(0).unsqueeze(0).float(),
                        size=(target_h, target_w), mode='nearest'
                    ).squeeze().bool().cpu().numpy()
                    gt_resized_np[inv_np] = 255

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
    CHECKPOINT_PATH = "/home/rvl1421/SAM_research-1/segment-anything/outputs_weather_sam_mask2former_testv3/weather_sam_best_latest.pth"
    TEST_CSV_PATH = "/home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_val.csv" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Model...")
    model = build_weather_sam_vit_h(checkpoint=CHECKPOINT_PATH)
    model.to(DEVICE)
    model.eval()

    test_ds = WeatherSegmentationDataset(csv_file=TEST_CSV_PATH, image_size=1024, mode='val')
    # 保留原作法：關閉 feature cache 以確保跑原始影像路徑；
    # 若測試 CSV 有 feature_path 且欲加速，可移除下一行。
    test_ds.has_cached_features = False
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn
    )

    runner = InferenceRunner(model, DEVICE, output_dir="inference_viz_acdc_testv3")
    runner.run_inference(test_loader, num_samples=None)