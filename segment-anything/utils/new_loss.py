import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=2.0, dice_weight=1.0, iou_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.label_smoothing = label_smoothing
        self.smooth = 1e-5
        
        # 真正的 Focal Loss 超參數
        self.gamma = 2.0
        self.alpha = 0.25 # 平衡前景(1)與背景(0)的初始權重

        self.class_map = {
            "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
            "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
            "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
            "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
        }

    def forward(self, pred_masks, gt_mask, iou_predictions, text_prompts):
        device = pred_masks.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {"total": 0.0, "focal": 0.0, "dice": 0.0, "iou_mse": 0.0}
        
        valid_prompts_count = 0

        # 1. 計算有效像素數 (必須轉 float 避免 FP16 溢位)
        valid_mask = (gt_mask != 255).float().unsqueeze(0).to(device)
        num_valid_pixels = valid_mask.sum(dim=(1, 2)) + 1e-6 
        
        # 防呆：如果整張圖都是 255 (無效)，直接回傳 0
        if valid_mask.sum() < 1:
            return total_loss, metrics

        for k, prompt in enumerate(text_prompts):
            if prompt not in self.class_map:
                continue
            
            valid_prompts_count += 1
            class_id = self.class_map[prompt]
            
            target = (gt_mask == class_id).float().unsqueeze(0).to(device)

            if self.label_smoothing > 0:
                target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            current_preds = pred_masks[k]
            # 限制範圍防止指數爆炸
            current_preds = torch.clamp(current_preds, min=-10.0, max=10.0)
            pred_prob = torch.sigmoid(current_preds)
            
            # ==========================================
            # 🟢 1. 計算 Dice Loss (針對前景結構)
            # ==========================================
            p_masked = pred_prob * valid_mask
            t_masked = target * valid_mask 
            
            intersection = (p_masked * t_masked).float().sum(dim=(1, 2)) 
            union = p_masked.float().sum(dim=(1, 2)) + t_masked.float().sum(dim=(1, 2))
            dice_losses = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            
            # ==========================================
            # 🔴 2. 真正的 Focal Loss (取代純 BCE)
            # ==========================================
            target_expanded = target.expand_as(current_preds)
            
            # 標準 BCE (不進行 reduction)
            bce_loss = F.binary_cross_entropy_with_logits(current_preds, target_expanded, reduction='none')
            
            # 計算預測正確的機率 p_t
            # 如果 target=1, pt = pred_prob; 如果 target=0, pt = 1 - pred_prob
            p_t = pred_prob * target_expanded + (1 - pred_prob) * (1 - target_expanded)
            
            # Focal 動態調節因子: (1 - p_t)^gamma (猜得越準，這個值越接近0，Loss越小)
            focal_term = (1.0 - p_t) ** self.gamma
            
            # Alpha 類別平衡因子 (前景用 alpha, 背景用 1-alpha)
            alpha_term = self.alpha * target_expanded + (1.0 - self.alpha) * (1.0 - target_expanded)
            
            # 組合真・Focal Loss
            focal_loss_pixel = alpha_term * focal_term * bce_loss
            masked_focal_loss = focal_loss_pixel * valid_mask
            
            # 平均到有效像素上
            focal_losses = masked_focal_loss.float().sum(dim=(1, 2)) / num_valid_pixels
            
            # ==========================================
            # 🔵 3. 組合 Loss 與 Min-Loss 策略
            # ==========================================
            seg_losses = (self.focal_weight * focal_losses) + (self.dice_weight * dice_losses)
            
            # SAM 的特色：選出預測最好的那一個 Mask 來算 Loss
            best_mask_idx = torch.argmin(seg_losses)
            min_seg_loss = seg_losses[best_mask_idx]
            
            # ==========================================
            # 🟡 4. IoU 預測 Loss (MSE)
            # ==========================================
            with torch.no_grad():
                pred_binary = (pred_prob[best_mask_idx] > 0.5).float()
                inter = (pred_binary * target.squeeze(0) * valid_mask.squeeze(0)).float().sum()
                uni = (pred_binary * valid_mask.squeeze(0)).float().sum() + \
                      (target.squeeze(0) * valid_mask.squeeze(0)).float().sum() - inter
                actual_iou = inter / (uni + 1e-5)
            
            pred_iou_conf = iou_predictions[k, best_mask_idx]
            iou_loss = F.mse_loss(pred_iou_conf, actual_iou)
            
            # 加總到此 Batch (Image) 的 Total Loss
            total_loss = total_loss + (min_seg_loss + self.iou_weight * iou_loss)
            
            metrics["focal"] += focal_losses[best_mask_idx].item()
            metrics["dice"] += dice_losses[best_mask_idx].item()
            metrics["iou_mse"] += iou_loss.item()

        # 正規化 Prompts 數量
        if valid_prompts_count > 0:
            total_loss = total_loss / valid_prompts_count
            for k in metrics:
                if k != "total":
                    metrics[k] /= valid_prompts_count

        metrics["total"] = total_loss.item()
        
        # 🛡️ 最終防線：捕捉 NaN/Inf，防止混合精度(AMP)訓練時導致 Scaler 崩潰
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠️ Warning: Loss became NaN/Inf! Returning zero loss to prevent crash.")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss, metrics