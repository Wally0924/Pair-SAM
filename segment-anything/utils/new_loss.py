import torch
import torch.nn as nn
import torch.nn.functional as F

# ACDC 訓練集實測 class pixel frequency（1200 張，2.3B valid pixels）
# 統計指令：pair_dataloader CSV → cv2 imread GT → np.bincount
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


# Cityscapes 訓練集實測 class pixel frequency（2975 張 labelTrainIds，5.521B valid pixels）
# 統計指令：cv2 imread labelTrainIds (GRAYSCALE) → np.bincount(minlength=256)[:19] → 正規化
# 與 ACDC 分佈差異大：road 0.369（ACDC 0.156）、sky 0.040（ACDC 0.337）、car 0.070（ACDC 0.019）。
# 供 Stage-1 Cityscapes encoder pretrain 使用，避免沿用 ACDC 頻率錯配稀有類梯度加權。
_CITYSCAPES_CLASS_FREQ = torch.tensor([
    0.368810,  # road
    0.060869,  # sidewalk
    0.228196,  # building
    0.006559,  # wall
    0.008783,  # fence  ← median
    0.012276,  # pole
    0.002085,  # traffic light
    0.005529,  # traffic sign
    0.159174,  # vegetation
    0.011587,  # terrain
    0.040115,  # sky
    0.012173,  # person
    0.001349,  # rider
    0.070011,  # car
    0.002676,  # truck
    0.002354,  # bus
    0.002330,  # train
    0.000986,  # motorcycle
    0.004139,  # bicycle
])

# Cityscapes 版 MFB 權重（同 sqrt-smoothed、cap=10、均值=1）
CITYSCAPES_CLASS_WEIGHTS: torch.Tensor = _build_median_freq_weights(_CITYSCAPES_CLASS_FREQ)


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


def lovasz_weight_for_epoch(epoch_index: int, start_epoch: int,
                            target_weight: float) -> float:
    """依當前 epoch 決定生效的 Lovász 權重（延後啟動排程）。

    前 start_epoch 個 epoch（epoch_index < start_epoch）停用 Lovász（回傳 0.0），
    讓 CE 先把分割學到合理水準，再於 epoch_index >= start_epoch 起加入 Lovász
    微調 mIoU —— 對應 Berman et al. (CVPR 2018) 原論文「先 CE 預訓、後接 Lovász
    fine-tune」的協定。start_epoch=0 → 從頭啟用，完全向後相容。

    Args:
        epoch_index:   0-based 的當前 epoch 索引。
        start_epoch:   Lovász 開始生效的 epoch 索引（前幾個 epoch 停用）。
        target_weight: 啟用後的目標權重（即 --lovasz_weight）。

    Returns:
        該 epoch 實際生效的 Lovász 權重。
    """
    return target_weight if epoch_index >= start_epoch else 0.0


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
                 label_smoothing: float = 0.0, lovasz_weight: float = 0.0,
                 use_mfb: bool = True, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce_weight     = ce_weight
        self.lovasz_weight = lovasz_weight
        self.use_mfb       = use_mfb

        # class_weights=None 時沿用 ACDC MFB（維持既有 ACDC trainer 行為）；
        # Stage-1 Cityscapes pretrain 傳入 CITYSCAPES_CLASS_WEIGHTS 以對齊其類別分佈。
        if class_weights is None:
            class_weights = _build_median_freq_weights(_ACDC_CLASS_FREQ)
        self.register_buffer('class_weights', class_weights)
        ce_weight_arg = self.class_weights if use_mfb else None  # [ablation] off = uniform
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=ce_weight_arg, ignore_index=255,
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
    計算 SAM Mask Decoder 輸出的候選 Mask 的 Dice Loss（K 個候選同時計算）。
    """
    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = 1e-5

    def forward(self, pred_masks: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor):
        """
        Inputs:
            pred_masks: (B, K, H, W) 原始 Mask Logits。
            target_mask: (B, 1, H, W) 二值化 Ground Truth。
            valid_mask: (B, 1, H, W) 過濾 255 忽略區。
        Returns:
            total_loss: (B, K) dice_weight × dice。
            dice_loss:  (B, K) 原始 Dice Loss。
        """
        p_t = torch.sigmoid(pred_masks).clamp(min=1e-7, max=1.0 - 1e-7)
        p_t_masked = p_t * valid_mask
        target_t_masked = target_mask * valid_mask
        target_t_expanded = target_t_masked.expand(-1, pred_masks.shape[1], -1, -1)

        intersection = (p_t_masked * target_t_masked).sum(dim=(2, 3))            # (B, K)
        union = p_t_masked.sum(dim=(2, 3)) + target_t_expanded.sum(dim=(2, 3))   # (B, K)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        total_loss = self.dice_weight * dice_loss
        return total_loss, dice_loss
