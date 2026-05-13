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


# 預算好的 MFB 類別權重（供 MaskLoss 加權平均使用）
# 稀少類（rider/motorcycle/bicycle）最高可達 cap=10，正規化後均值=1
ACDC_CLASS_WEIGHTS: torch.Tensor = _build_median_freq_weights(_ACDC_CLASS_FREQ)


# =============================================================================
# Lovász-Softmax Loss（嵌入實作，無需外部套件）
# 來源：Berman et al., CVPR 2018 — "The Lovász-Softmax Loss"
# https://github.com/bermanmaxim/LovaszSoftmax
# 直接優化 IoU 的可微近似，梯度方向與 mIoU 評估指標完全對齊。
# =============================================================================

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """計算 Lovász extension 的梯度（排序後的 GT 向量）。"""
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:p - 1]
    return jaccard


def _lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor,
                          classes: str = 'present') -> torch.Tensor:
    """
    probs : (P, C) softmax 機率，P = 有效像素數
    labels: (P,)   GT 類別 id
    classes: 'all' 計算全部 C 類；'present' 只計算出現在此 batch 的類別
    """
    C = probs.shape[1]
    losses = []
    class_to_sum = list(range(C)) if classes == 'all' else torch.unique(labels).tolist()
    for c in class_to_sum:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor,
                   classes: str = 'present', ignore_index: int = 255) -> torch.Tensor:
    """
    多類別 Lovász-Softmax Loss。

    Inputs:
        probs : (B, C, H, W) softmax 機率（請先對 logits 做 .softmax(dim=1)）
        labels: (B, H, W)    GT（ignore_index 會被過濾）
        classes: 'present' 僅計算 batch 中出現的類別（訓練用，更穩定）
        ignore_index: 忽略的 GT 值（預設 255）
    """
    B, C, H, W = probs.shape
    probs_flat  = probs.permute(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, C)
    labels_flat = labels.reshape(-1)                           # (B*H*W,)

    valid = labels_flat != ignore_index
    probs_flat  = probs_flat[valid]
    labels_flat = labels_flat[valid]

    if labels_flat.numel() == 0:
        return probs.sum() * 0.0

    return _lovasz_softmax_flat(probs_flat, labels_flat, classes=classes)


# =============================================================================
# ContextLoss
# =============================================================================

class ContextLoss(nn.Module):
    """
    計算 ResidualDWConvFusion 輸出的 19 類別高解析度 logits 的語意分割損失。

    組成：
      - weighted CE（Median Frequency Balancing）：補償 long-tail class 監督信號
      - Lovász-Softmax（可選）：直接優化 mIoU 的可微近似，梯度方向與評估指標對齊

    lovasz_weight=0.0 時退化為原始純 CE 行為，完全向後相容。
    """
    def __init__(self, ce_weight: float = 1.0, num_classes: int = 19,
                 label_smoothing: float = 0.0, lovasz_weight: float = 0.0):
        super().__init__()
        self.ce_weight     = ce_weight
        self.lovasz_weight = lovasz_weight

        class_weights = _build_median_freq_weights(_ACDC_CLASS_FREQ)
        self.register_buffer('class_weights', class_weights)
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=255,
            label_smoothing=label_smoothing,
        )
        self.ce_unweighted = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, fused_logits_hr: torch.Tensor, gt_mask: torch.Tensor):
        """
        Inputs:
            fused_logits_hr: (B, 19, H, W) 融合後的 19 類別高解析度特徵圖。
            gt_mask: (B, H, W) 標註的 Ground Truth (0~18 類別，255 為忽略區)。
        Returns:
            total_loss:   (scalar) 加權總損失（用於 backward）。
            ce_val:       (float)  未加權 CE（純語意難度監控）。
            lov_val:      (float)  Lovász-Softmax loss（lovasz_weight=0 時為 0）。
            ce_weighted_val: (float) MFB 加權 CE（與 total_loss 中實際使用的一致；
                              監控稀有類懲罰是否爆炸）。
        """
        valid_pixel_count = (gt_mask != 255).sum()
        if valid_pixel_count == 0:
            zero = torch.zeros((), device=fused_logits_hr.device, dtype=fused_logits_hr.dtype)
            return zero, 0.0, 0.0, 0.0

        # ── CE：weighted（進 total）+ unweighted（純監控）──
        ce_loss = self.ce_loss_fn(fused_logits_hr, gt_mask)             # MFB-weighted
        ce_val  = self.ce_unweighted(fused_logits_hr, gt_mask).item()    # unweighted
        ce_weighted_val = float(ce_loss.item())                          # weighted scalar

        total   = self.ce_weight * ce_loss
        lov_val = 0.0

        # ── Lovász-Softmax（lovasz_weight=0 時跳過，不影響原有行為）──
        if self.lovasz_weight > 0.0:
            probs    = fused_logits_hr.softmax(dim=1)
            lov_loss = lovasz_softmax(probs, gt_mask, classes='present', ignore_index=255)
            total    = total + self.lovasz_weight * lov_loss
            lov_val  = lov_loss.item()

        return total, ce_val, lov_val, ce_weighted_val



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
