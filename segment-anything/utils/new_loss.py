import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=2.0, dice_weight=1.0, iou_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.label_smoothing = label_smoothing  # Label Smoothing 參數
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
        修正版: 實作 Min-Loss Strategy + 忽略 Ignore Label (255)
        """
        device = pred_masks.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {"total": 0.0, "bce": 0.0, "dice": 0.0, "iou_mse": 0.0}
        
        valid_prompts_count = 0

        # [新增 1] 建立有效區域遮罩 (Valid Mask)
        # gt_mask shape: (H, W) -> 數值為 0~18 或 255
        # valid_mask: (1, H, W) -> 255的地方是 0 (False), 其他是 1 (True)
        valid_mask = (gt_mask != 255).float().unsqueeze(0).to(device)

        num_valid_pixels_raw = valid_mask.sum(dim=(1, 2))
        
        if num_valid_pixels_raw < 1:
                    return total_loss, metrics
        
        # 避免除以 0，加一個極小值
        num_valid_pixels = valid_mask.sum(dim=(1, 2)) + 1e-6 

        for k, prompt in enumerate(text_prompts):
            if prompt not in self.class_map:
                continue
            
            valid_prompts_count += 1
            class_id = self.class_map[prompt]
            
            # 1. 準備 GT: (1, H, W)
            # 在 255 的區域，target 會變成 0 (因為 255 != class_id)，這沒問題
            target = (gt_mask == class_id).float().unsqueeze(0).to(device)

            # [新增步驟] 實作 Label Smoothing
            # 公式: New_Target = Original_Target * (1 - epsilon) + epsilon / 2
            # 針對 Binary Classification (0/1) 的平滑方式
            if self.label_smoothing > 0:
                target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # 2. 取出預測: (3, H, W)
            current_preds = pred_masks[k]

            current_preds = torch.clamp(current_preds, min=-10.0, max=10.0)
            
            # --- 計算 Dice Loss (需過濾無效區域) ---
            pred_prob = torch.sigmoid(current_preds)
            
            # [修改 2] 在計算 Intersection 和 Union 前，先過濾掉無效區域
            # 這樣如果模型在白色區塊預測了東西，也不會被懲罰
            p_masked = pred_prob * valid_mask
            t_masked = target * valid_mask # 雖然 target 在無效區本來就是 0，但乘一下保險
            
            intersection = (p_masked * t_masked).sum(dim=(1, 2)) 
            union = p_masked.sum(dim=(1, 2)) + t_masked.sum(dim=(1, 2))
            dice_losses = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            
            # --- 計算 Focal / BCE Loss (需過濾無效區域) ---
            target_expanded = target.expand_as(current_preds)
            
            # [修改 3] 使用 reduction='none' 算出每個像素的 Loss
            pixel_loss = F.binary_cross_entropy_with_logits(
                current_preds, target_expanded, reduction='none'
            )
            
            # [修改 4] 乘上 valid_mask，把 255 區域的 Loss 變成 0
            # pixel_loss: (3, H, W), valid_mask: (1, H, W) -> 廣播相乘
            masked_pixel_loss = pixel_loss * valid_mask
            
            # [修改 5] 取平均時，分母要是 "有效像素數量"，而不是 "總像素數量"
            bce_losses = masked_pixel_loss.sum(dim=(1, 2)) / num_valid_pixels
            
            # 若您有使用 Focal Weight 或 Pixel Weight，乘法操作要在 sum 之前做
            # bce_losses = (masked_pixel_loss * pixel_weight).sum(...) / num_valid_pixels
            
            # --- 組合 Loss ---
            seg_losses = (self.focal_weight * bce_losses) + (self.dice_weight * dice_losses)
            
            # Min-Loss Strategy
            best_mask_idx = torch.argmin(seg_losses)
            min_seg_loss = seg_losses[best_mask_idx]
            
            # --- IoU Loss ---
            with torch.no_grad():
                # 計算真實 IoU 時也要考慮 valid_mask
                pred_binary = (pred_prob[best_mask_idx] > 0.5).float()
                # 這裡也要乘 valid_mask
                inter = (pred_binary * target.squeeze(0) * valid_mask.squeeze(0)).sum()
                uni = (pred_binary * valid_mask.squeeze(0)).sum() + (target.squeeze(0) * valid_mask.squeeze(0)).sum() - inter
                actual_iou = inter / (uni + 1e-5)
            
            pred_iou_conf = iou_predictions[k, best_mask_idx]
            iou_loss = F.mse_loss(pred_iou_conf, actual_iou)
            
            total_loss = total_loss + (min_seg_loss + self.iou_weight * iou_loss)
            
            # Metrics
            metrics["bce"] += bce_losses[best_mask_idx].item()
            metrics["dice"] += dice_losses[best_mask_idx].item()
            metrics["iou_mse"] += iou_loss.item()

        if valid_prompts_count > 0:
            total_loss = total_loss / valid_prompts_count
            for k in metrics:
                if k != "total":
                    metrics[k] /= valid_prompts_count

        metrics["total"] = total_loss.item()
        
        return total_loss, metrics
