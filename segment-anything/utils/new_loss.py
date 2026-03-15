import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextLoss(nn.Module):
    """
    計算 ContextFusionHead 輸出的 19 類別特徵圖的 Cross Entropy Loss。
    用於監督模型學習全域的空間佈局與類別互斥性。
    """
    def __init__(self, ce_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, fused_logits_hr: torch.Tensor, gt_mask: torch.Tensor):
        """
        Inputs:
            fused_logits_hr: (B, 19, H, W) 融合後的 19 類別高解析度特徵圖。
            gt_mask: (B, H, W) 標註的 Ground Truth (0~18 類別，255 為忽略區)。
        Returns:
            total_loss: (scalar) 加權後的 Cross Entropy Loss。
            ce_val: (float) 原始未加權的 Cross Entropy Loss 數值。
        """
        ce_loss = self.ce_loss_fn(fused_logits_hr, gt_mask)
        total_loss = self.ce_weight * ce_loss
        return total_loss, ce_loss.item()


class MaskLoss(nn.Module):
    """
    計算 SAM Mask Decoder 輸出的候選 Mask 的 Focal Loss 與 Dice Loss。
    支援同時計算 K 個候選 Mask，並用於後續選取 Minimum Loss。
    """
    def __init__(self, focal_weight: float = 20.0, dice_weight: float = 1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-5
        
        self.gamma = 2.0
        self.alpha = 0.25 

    def forward(self, pred_masks: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor):
        """
        Inputs:
            pred_masks: (B, K, H, W) 原始的 Mask Logits (例如 3 個候選)。
            target_mask: (B, 1, H, W) 針對特定單一類別的二值化 Ground Truth。
            valid_mask: (B, 1, H, W) 過濾掉 255 忽略區域的遮罩。
        Returns:
            total_loss: (B, K) K 個候選 Mask的加權總和 Loss。
            focal_loss: (B, K) K 個候選 Mask 的 Focal Loss。
            dice_loss: (B, K) K 個候選 Mask 的 Dice Loss。
        """
        B, K, H, W = pred_masks.shape
        
        p_t = torch.sigmoid(pred_masks)
        
        p_t_masked = p_t * valid_mask
        target_t_masked = target_mask * valid_mask
        
        # --- 計算 Focal Loss ---
        p_t_clamped = torch.clamp(p_t, min=1e-7, max=1.0-1e-7)
        p_t_clamped_masked = p_t_clamped * valid_mask
        prob_correct = p_t_clamped_masked * target_t_masked + (1.0 - p_t_clamped) * (1.0 - target_t_masked) * valid_mask
        
        target_t_expanded = target_mask.expand(-1, K, -1, -1)
        bce_loss_raw = F.binary_cross_entropy_with_logits(pred_masks, target_t_expanded, reduction='none')
        bce_loss = bce_loss_raw * valid_mask
        
        focal_term = (1.0 - prob_correct) ** self.gamma
        alpha_term = self.alpha * target_t_masked + (1.0 - self.alpha) * (1.0 - target_t_masked)
        
        focal_loss_pixel = alpha_term * focal_term * bce_loss
        
        valid_pixels = valid_mask.sum(dim=(2, 3)) + 1e-6 # (B, 1)
        focal_loss = focal_loss_pixel.sum(dim=(2, 3)) / valid_pixels # (B, K)
        
        # --- 計算 Dice Loss ---
        intersection = (p_t_clamped_masked * target_t_masked).sum(dim=(2, 3)) # (B, K)
        target_t_masked_expanded = target_t_masked.expand(-1, K, -1, -1) # (B, K, H, W)
        union = p_t_clamped_masked.sum(dim=(2, 3)) + target_t_masked_expanded.sum(dim=(2, 3)) # (B, K)
        
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth) # (B, K)
        
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        return total_loss, focal_loss, dice_loss


def calculate_true_iou(pred_masks: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    計算二值化預測圖與 Ground Truth 之間的真實交併比 (True IoU)。
    用於監督 IoU Prediction Head 的準確度。
    
    Inputs:
        pred_masks: (B, K, H, W) 原始的 Mask Logits。
        target_mask: (B, 1, H, W) 二值化 Ground Truth。
        valid_mask: (B, 1, H, W) 過濾掉 255 忽略區域的遮罩。
    Returns:
        iou: (B, K) 給定 K 個候選的 True IoU 數值。
    """
    pred_binary = (pred_masks > 0.0).float() * valid_mask
    target_masked = target_mask * valid_mask
    
    intersection = (pred_binary * target_masked).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + target_masked.sum(dim=(2, 3)) - intersection
    
    iou = intersection / (union + 1e-6)
    return iou