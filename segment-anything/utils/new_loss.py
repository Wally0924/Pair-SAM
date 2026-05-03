import torch
import torch.nn as nn
import torch.nn.functional as F

# ACDC 訓練集實測 class pixel frequency（1200 張，2.3B valid pixels）
# 統計指令：weather_dataloader CSV → cv2 imread GT → np.bincount
# sky 佔 33.7%（ACDC 特有，遠高於 Cityscapes 的 7%），road/building/vegetation 次之
# rider/motorcycle/bicycle 低於 0.05%，觸及 weight cap
_ACDC_CLASS_FREQ = torch.tensor([
    0.155722,  # road
    0.046873,  # sidewalk
    0.172310,  # building
    0.013243,  # wall
    0.016884,  # fence
    0.010085,  # pole  ← median
    0.001909,  # traffic light
    0.003246,  # traffic sign
    0.194371,  # vegetation
    0.017808,  # terrain
    0.336727,  # sky
    0.001106,  # person
    0.000133,  # rider
    0.019480,  # car
    0.003907,  # truck
    0.002188,  # bus
    0.003293,  # train
    0.000238,  # motorcycle
    0.000476,  # bicycle
])

def _build_median_freq_weights(freq: torch.Tensor, cap: float = 10.0) -> torch.Tensor:
    """Square-root smoothed median frequency balancing。
    sqrt 壓縮極端比值，避免 sky/vegetation 等大面積類別梯度貢獻過低，
    同時保留 long-tail 類別的高權重特性，再正規化使均值=1。
    """
    median = freq.median()
    raw_weights = torch.sqrt(median / freq.clamp(min=1e-8))
    w = raw_weights.clamp(max=cap)
    return w / w.mean()


class ContextLoss(nn.Module):
    """
    計算 ResidualDWConvFusion 輸出的 19 類別高解析度 logits 的 Cross Entropy Loss。
    使用 ACDC 訓練集實測頻率做 Median Frequency Balancing，補償 long-tail class 監督信號。
    """
    def __init__(self, ce_weight: float = 1.0, num_classes: int = 19):
        super().__init__()
        self.ce_weight = ce_weight
        class_weights = _build_median_freq_weights(_ACDC_CLASS_FREQ)
        # register_buffer 確保 .to(device) 時自動跟著移動
        self.register_buffer('class_weights', class_weights)
        self.ce_loss_fn   = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)
        self.ce_unweighted = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, fused_logits_hr: torch.Tensor, gt_mask: torch.Tensor):
        """
        Inputs:
            fused_logits_hr: (B, 19, H, W) 融合後的 19 類別高解析度特徵圖。
            gt_mask: (B, H, W) 標註的 Ground Truth (0~18 類別，255 為忽略區)。
        Returns:
            total_loss: (scalar) weighted CE loss（用於 backward）。
            ce_val: (float) 原始未加權 CE（用於 monitoring，與舊版 log 可比較）。
        """
        valid_pixel_count = (gt_mask != 255).sum()
        if valid_pixel_count == 0:
            zero = torch.zeros((), device=fused_logits_hr.device, dtype=fused_logits_hr.dtype)
            return zero, 0.0

        ce_loss = self.ce_loss_fn(fused_logits_hr, gt_mask)
        ce_val  = self.ce_unweighted(fused_logits_hr, gt_mask).item()
        return self.ce_weight * ce_loss, ce_val



class MaskLoss(nn.Module):
    """
    計算 SAM Mask Decoder 輸出的候選 Mask 的 Focal Loss 與 Dice Loss。
    支援同時計算 K 個候選 Mask，並用於後續選取 Minimum Loss。
    """
    def __init__(self, focal_weight: float = 5.0, dice_weight: float = 1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-5
        
        self.gamma = 2.0
        self.alpha = 0.75  # 正樣本（前景）權重，負樣本為 1-alpha=0.25；必須在 (0,1)

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


