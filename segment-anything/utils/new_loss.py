import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=2.0, dice_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.smooth = 1e-05 # 數值穩定與平滑

        # 定義類別映射表 (必須與 dataloader 一致)
        self.class_map = {
            "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
            "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
            "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
            "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
        }

    def forward(self, pred_masks, gt_mask, iou_predictions, text_prompts):
        """
        修正版: 實作 Min-Loss Strategy (Anchor-Free like)
        
        Args:
            pred_masks: (K, 3, H, W) - K個Prompts, 每個有3個多義性遮罩
            gt_mask: (H, W) - 真實標籤
            iou_predictions: (K, 3) - 預測的 IoU 信心度
        """
        device = pred_masks.device
        total_loss = 0.0
        metrics = {"total": 0.0, "bce": 0.0, "dice": 0.0, "iou_mse": 0.0}
        
        valid_prompts_count = 0

        # 針對每一個 Prompt 獨立計算
        for k, prompt in enumerate(text_prompts):
            if prompt not in self.class_map:
                continue
            
            valid_prompts_count += 1
            class_id = self.class_map[prompt]
            
            # 1. 準備 GT: (1, H, W)
            target = (gt_mask == class_id).float().unsqueeze(0).to(device)
            
            # 2. 取出該 Prompt 的 3 個預測遮罩: (3, H, W)
            # 這裡我們不看 iou_predictions 選誰，而是三個都算
            current_preds = pred_masks[k] # (3, H, W)
            
            # 為了廣播計算，擴展維度
            # Preds: (3, H, W), Target: (1, H, W) -> Broadcasting OK
            
            # --- 計算每個 Mask 的 Dice Loss ---
            pred_prob = torch.sigmoid(current_preds)
            intersection = (pred_prob * target).sum(dim=(1, 2)) # (3,)
            union = pred_prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) # (3,)
            dice_losses = 1 - (2 * intersection + self.smooth) / (union + self.smooth) # (3,)
            
            # --- 計算每個 Mask 的 Focal Loss ---
            # 這裡需要把 target 擴展成 (3, H, W)
            target_expanded = target.expand_as(current_preds)
            bce_losses = F.binary_cross_entropy_with_logits(
                current_preds, target_expanded, reduction='none'
            ).mean(dim=(1, 2)) # (3,)
            
            # --- 組合分割 Loss ---
            # 這裡暫時不加 IoU Loss，只用分割品質來決定誰是最好的 Mask
            seg_losses = (self.focal_weight * bce_losses) + (self.dice_weight * dice_losses)
            
            # 3. [關鍵] Min-Loss Strategy
            # 找出 Loss 最小的那個 Mask index (0, 1, or 2)
            best_mask_idx = torch.argmin(seg_losses)
            
            min_seg_loss = seg_losses[best_mask_idx]
            best_dice_val = dice_losses[best_mask_idx]
            best_bce_val = bce_losses[best_mask_idx]
            
            # 4. IoU Head 的監督
            # 我們希望模型預測的 IoU 分數，能逼近「真實計算出來的 IoU」
            # 只有「最佳 Mask」對應的 IoU Head 需要被更新，其他的不管 (Hard Example Mining 概念)
            with torch.no_grad():
                pred_binary = (pred_prob[best_mask_idx] > 0.5).float()
                inter = (pred_binary * target.squeeze(0)).sum()
                uni = pred_binary.sum() + target.squeeze(0).sum() - inter
                actual_iou = inter / (uni + 1e-5)
                
            pred_iou_conf = iou_predictions[k, best_mask_idx]
            iou_loss = F.mse_loss(pred_iou_conf, actual_iou)
            
            # 5. 該 Prompt 的總 Loss
            prompt_total_loss = min_seg_loss + (self.iou_weight * iou_loss)
            
            total_loss += prompt_total_loss
            metrics["bce"] += best_bce_val.item()
            metrics["dice"] += best_dice_val.item()
            metrics["iou_mse"] += iou_loss.item()

        if valid_prompts_count > 0:
            total_loss /= valid_prompts_count
            for k in metrics:
                metrics[k] /= valid_prompts_count
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        metrics["total"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, metrics

#嘗試一下根據不同物件給予不同的 Loss 權重
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SAMLoss(nn.Module):
#     def __init__(self, focal_weight=2.0, dice_weight=2.0, iou_weight=1.0):
#         """
#         參數設定說明:
#         focal_weight=2.0 : 懲罰背景雜訊 (對於霧天影像很重要)
#         dice_weight=2.0  : 強調物件形狀的完整性 (避免斷裂)
#         iou_weight=1.0   : 訓練信心度預測頭
#         """
#         super().__init__()
#         self.focal_weight = focal_weight
#         self.dice_weight = dice_weight
#         self.iou_weight = iou_weight
#         self.smooth = 1e-5

#         # 類別映射 (需與 weather_dataloader.py 一致)
#         self.class_map = {
#             "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
#             "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
#             "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
#             "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
#         }
        
#         # [核心修改 1] Class Boosting: 針對難以辨識的小物件給予倍率懲罰
#         # 由於 ViT 被凍結，它在霧中幾乎看不到桿子。
#         # 我們必須用極大的 Loss (x5.0) 強迫 Fusion Module 去"相信" Reference Mask 提供的桿子位置。
#         self.class_weights = {
#             "pole": 5.0, "traffic light": 5.0, "traffic sign": 5.0, # 極細小
#             "person": 4.0, "rider": 4.0, "bicycle": 4.0, "motorcycle": 4.0, # 中小型
#             "wall": 2.0, "fence": 2.0, # 邊界模糊
#             # 大物件維持 1.0
#             "road": 1.0, "building": 1.0, "vegetation": 1.0, "sky": 1.0, 
#             "sidewalk": 1.0, "car": 1.0, "truck": 1.0, "bus": 1.0, "train": 1.0
#         }

#     def forward(self, pred_masks, gt_mask, iou_predictions, text_prompts):
#         device = pred_masks.device
#         total_loss = 0.0
#         metrics = {"total": 0.0, "bce": 0.0, "dice": 0.0, "iou_mse": 0.0}
        
#         valid_prompts_count = 0

#         for k, prompt in enumerate(text_prompts):
#             if prompt not in self.class_map:
#                 continue
            
#             valid_prompts_count += 1
#             class_id = self.class_map[prompt]
#             cls_weight = self.class_weights.get(prompt, 1.0)

#             # 準備 GT: (1, H, W)
#             target = (gt_mask == class_id).float().unsqueeze(0).to(device)
            
#             # 取出預測: (3, H, W)
#             current_preds = pred_masks[k] 
            
#             # --- Dice Loss ---
#             pred_prob = torch.sigmoid(current_preds)
#             intersection = (pred_prob * target).sum(dim=(1, 2))
#             union = pred_prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
#             dice_losses = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            
#             # --- Weighted Focal Loss ---
#             target_expanded = target.expand_as(current_preds)
            
#             # [核心修改 2] Pixel-wise Weighting (像素聚焦)
#             # 在同一張圖中，前景(物件)像素的權重是背景的 10 倍 + 1
#             # 這能防止小物件因為像素太少(佔畫面 0.01%)而被 Loss 忽略
#             pixel_weight = target_expanded * 10.0 + 1.0 
            
#             bce_losses = F.binary_cross_entropy_with_logits(
#                 current_preds, target_expanded, weight=pixel_weight, reduction='none'
#             ).mean(dim=(1, 2))
            
#             # --- 組合 Loss ---
#             # 這裡應用了您的設定 (Focal=2.0, Dice=2.0) 並乘上 Class Weight
#             base_seg_loss = (self.focal_weight * bce_losses) + (self.dice_weight * dice_losses)
#             seg_losses = cls_weight * base_seg_loss
            
#             # Min-Loss Strategy (選擇最好的那一個 Mask 來更新)
#             best_mask_idx = torch.argmin(seg_losses)
#             min_seg_loss = seg_losses[best_mask_idx]
            
#             # --- IoU Loss ---
#             with torch.no_grad():
#                 pred_binary = (pred_prob[best_mask_idx] > 0.5).float()
#                 inter = (pred_binary * target.squeeze(0)).sum()
#                 uni = pred_binary.sum() + target.squeeze(0).sum() - inter
#                 actual_iou = inter / (uni + 1e-5)
            
#             pred_iou_conf = iou_predictions[k, best_mask_idx]
#             iou_loss = F.mse_loss(pred_iou_conf, actual_iou)
            
#             total_loss += (min_seg_loss + self.iou_weight * iou_loss)
            
#             # 紀錄
#             metrics["bce"] += bce_losses[best_mask_idx].item()
#             metrics["dice"] += dice_losses[best_mask_idx].item()
#             metrics["iou_mse"] += iou_loss.item()

#         if valid_prompts_count > 0:
#             total_loss /= valid_prompts_count
#             for k in metrics:
#                 metrics[k] /= valid_prompts_count
#         else:
#             total_loss = torch.tensor(0.0, device=device, requires_grad=True)

#         metrics["total"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
#         return total_loss, metrics