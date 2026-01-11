import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=20.0, dice_weight=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred_masks, gt_masks, iou_predictions):
        # 1. Dice Loss
        pred_prob = torch.sigmoid(pred_masks)
        intersection = (pred_prob * gt_masks).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        # 2. Focal Loss (BCE)
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        
        # 3. IoU prediction Loss (修正維度對齊)
        with torch.no_grad():
            pred_binary = (pred_prob > 0.5).float()
            inter = (pred_binary * gt_masks).sum(dim=(2, 3))
            uni = pred_binary.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) - inter
            actual_iou = inter / (uni + 1e-5) # Shape: (B, 1)

        # 確保預測值與實際值的 Shape 一致
        iou_loss = F.mse_loss(iou_predictions.view(-1), actual_iou.view(-1))
        
        # 總 Loss
        total_loss = (self.focal_weight * bce_loss) + (self.dice_weight * dice_loss) + iou_loss
        
        return total_loss, {
            "total": total_loss.item(),
            "bce": bce_loss.item(),
            "dice": dice_loss.item(),
            "iou_mse": iou_loss.item()
        }