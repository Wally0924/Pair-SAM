import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=2.0, dice_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.smooth = 1.0 # 數值穩定與平滑

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