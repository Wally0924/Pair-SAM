import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=20.0, dice_weight=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred_masks, gt_masks, iou_predictions):
        """
        Args:
            pred_masks: (B, 1, H, W) - Logits
            gt_masks: (B, 1, H, W) - 0 or 1
            iou_predictions: (B, 1) - Predicted IoU scores
        """
        # 1. Dice Loss
        pred_prob = torch.sigmoid(pred_masks)
        intersection = (pred_prob * gt_masks).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        # 2. Focal Loss (Binary Cross Entropy with sigmoid built-in)
        # 這裡使用簡單的 BCE 作為 Focal Loss 的近似或基礎，也可實作完整的 Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        
        # 3. IoU prediction Loss (MSE between predicted IoU and actual IoU)
        # 計算實際 IoU 用於監督 IoU Head
        with torch.no_grad():
            pred_binary = (pred_prob > 0.5).float()
            inter = (pred_binary * gt_masks).sum(dim=(2, 3))
            uni = pred_binary.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) - inter
            actual_iou = (inter + 1e-5) / (uni + 1e-5)
        
        iou_loss = F.mse_loss(iou_predictions.flatten(), actual_iou.flatten())

        # 組合 Loss
        total_loss = self.focal_weight * bce_loss + self.dice_weight * dice_loss + iou_loss
        
        return total_loss, {"bce": bce_loss.item(), "dice": dice_loss.item(), "iou": iou_loss.item()}