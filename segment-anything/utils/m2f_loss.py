# ============================================================================
# Vendored from:
#   facebookresearch/Mask2Former
#     File: mask2former/modeling/criterion.py
#       (dice_loss, sigmoid_ce_loss, calculate_uncertainty,
#        SetCriterion.loss_labels / loss_masks 計算主體)
#     Commit: 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a
#     License: MIT (Copyright (c) Facebook, Inc. and its affiliates.)
#     Paper: Cheng et al., "Masked-attention Mask Transformer for Universal
#       Image Segmentation" (Mask2Former), CVPR 2022. arXiv:2112.01527
#   facebookresearch/detectron2
#     File: projects/PointRend/point_rend/point_features.py
#       (point_sample, get_uncertain_point_coords_with_randomness)
#     Commit: 02b5c4e295e990042a714712c21dc79b731e8833
#     License: Apache-2.0 (Copyright (c) Facebook, Inc. and its affiliates.)
#     Paper: Kirillov et al., "PointRend: Image Segmentation as Rendering",
#       CVPR 2020. arXiv:1912.08193
#   (Both commit hashes reused verbatim from Task 1 / Task 2's recorded
#    `git ls-remote HEAD` — re-checked at Task 3 time and confirmed unchanged.)
#
# [WeatherSAM adaptations]（完整清單；其餘計算式逐行同上游）:
#   1. HungarianMatcher 不移植，以 _fixed_match_targets 取代：query i ↔ 類別 i
#      硬對應，影像中缺席的類別 target = no-object。依據：OV-DETR conditional
#      matching（Zang et al., "Open-Vocabulary DETR with Conditional Matching",
#      ECCV 2022）— query 已帶類別語意時匹配恆等，不需要自由的二分圖匹配。
#   2. dice_loss / sigmoid_ce_loss 加 optional `weights` 參數（逐點 validity
#      權重）。weights=None 時計算式與上游完全相同。用於排除 ignore(255) 區域
#      的採樣點——上游 Cityscapes 語意訓練在 dataset mapper 階段就把 ignore
#      區從 target 剔除，本專案 GT 直接帶 255 + 逐點 validity mask，故把排除
#      動作移到點採樣之後（點層級）而非資料前處理層級。
#   3. 去掉 torch.jit.script 包裝（dice_loss_jit / sigmoid_ce_loss_jit）與
#      分散式 num_masks 歸一（get_world_size — 本專案單卡訓練）。
#      num_masks = 本張影像中 present 的類別數（等同 batch=1 時上游的
#      sum(len(t["labels"]) for t in targets)）。
#   4. detectron2.layers.cat（處理空 list 邊界的 torch.cat 薄封裝）替換為
#      torch.cat：本專案不依賴 detectron2，且呼叫處恆非空 list，行為等價。
#   5. 包裝類 M2FSetLoss.forward(output, gt_mask) 對齊 WeatherSAM trainer 的
#      per-image（B=1）呼叫慣例；deep supervision 迴圈對應上游 SetCriterion
#      .forward 中 aux_outputs 逐組同權重累加的邏輯。
# ============================================================================
import torch
import torch.nn.functional as F
from torch import nn


# ── point_sample / get_uncertain_point_coords_with_randomness：
#    自 detectron2 projects/PointRend/point_rend/point_features.py 原樣移植
#    （官方簽名、docstring、計算式逐行保留；detectron2.layers.cat →
#    torch.cat，見 adaptation 4）──


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(  # [WeatherSAM adaptation 4] detectron2.layers.cat -> torch.cat
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


# ── dice_loss / sigmoid_ce_loss / calculate_uncertainty：
#    自 mask2former/modeling/criterion.py 原樣移植（weights=None 時與上游
#    逐行同計算式）+ adaptation 2 的 optional weights 參數 ──


def dice_loss(inputs, targets, num_masks, weights=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        weights: [WeatherSAM adaptation 2] optional float tensor, same shape as
                 inputs, per-point validity weight (1.0 valid / 0.0 ignored).
                 None (default) reproduces upstream behaviour exactly.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    if weights is not None:                      # [WeatherSAM adaptation 2]
        inputs, targets = inputs * weights, targets * weights
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks, weights=None):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        weights: [WeatherSAM adaptation 2] optional float tensor, same shape as
                 inputs, per-point validity weight (1.0 valid / 0.0 ignored).
                 None (default) reproduces upstream behaviour exactly.
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if weights is not None:                      # [WeatherSAM adaptation 2]
        loss = (loss * weights).sum(1) / weights.sum(1).clamp(min=1.0)
    else:
        loss = loss.mean(1)
    return loss.sum() / num_masks


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class M2FSetLoss(nn.Module):
    """SetCriterion 的固定匹配包裝（見檔頭 adaptation 清單）。

    Consumes M2FDecoder（Task 2）輸出的
    {'pred_logits': (1,C,K+1), 'pred_masks': (1,C,H,W), 'aux_outputs': [...]}
    與 gt_mask (1,H_img,W_img) long（255=ignore），回傳
    (scalar_loss, {'cls','bce','dice'})（deep-supervision 主輸出的分量，
    純供 logging，實際 loss 已累加全部層）。
    """

    def __init__(self, num_classes=19, cls_weight=2.0, bce_weight=5.0,
                 dice_weight=5.0, no_object_weight=0.1, num_points=12544,
                 oversample_ratio=3.0, importance_sample_ratio=0.75,
                 ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        # 上游 SetCriterion.__init__ 的 empty_weight（eos_coef=no_object_weight）
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    def _fixed_match_targets(self, gt_mask):
        """[WeatherSAM adaptation 1] 固定匹配 target 建構（取代 HungarianMatcher）。"""
        valid = gt_mask != self.ignore_index
        labels = torch.full((self.num_classes,), self.num_classes,
                             dtype=torch.long, device=gt_mask.device)
        present, tgt_list = [], []
        for c in range(self.num_classes):
            m = (gt_mask == c) & valid
            if m.any():
                labels[c] = c
                present.append(c)
                tgt_list.append(m[0].float())
        tgt = torch.stack(tgt_list, 0).unsqueeze(1) if present else None  # (P,1,H,W)
        return labels, present, tgt, valid.float().unsqueeze(0)           # (1,1,H,W)

    def _loss_labels(self, out, labels):
        # 上游 SetCriterion.loss_labels 主體（B=1、固定匹配 → target_classes 即 labels）
        src_logits = out["pred_logits"][0].float()
        return F.cross_entropy(src_logits, labels, self.empty_weight)

    def _loss_masks(self, out, present, tgt, valid_f):
        # 上游 SetCriterion.loss_masks 主體：no_grad 點採樣 → point_sample → 兩個 loss
        if not present:
            zero = out["pred_masks"].sum() * 0.0
            return zero, zero
        src_masks = out["pred_masks"][0, present].unsqueeze(1).float()  # (P,1,256,256)
        num_masks = float(len(present))
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, lambda logits: calculate_uncertainty(logits),
                self.num_points, self.oversample_ratio, self.importance_sample_ratio,
            )
            point_labels = point_sample(tgt, point_coords, align_corners=False).squeeze(1)
            # [WeatherSAM adaptation 2] 逐點 validity（ignore 區域排除）
            point_valid = (point_sample(
                valid_f.expand(len(present), -1, -1, -1), point_coords,
                align_corners=False).squeeze(1) > 0.5).float()
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        l_bce = sigmoid_ce_loss(point_logits, point_labels, num_masks, weights=point_valid)
        l_dice = dice_loss(point_logits, point_labels, num_masks, weights=point_valid)
        return l_bce, l_dice

    def forward(self, output, gt_mask):
        labels, present, tgt, valid_f = self._fixed_match_targets(gt_mask)
        total, log = None, {}
        # 上游 SetCriterion.forward：主輸出 + aux_outputs 逐組同權重累加（deep supervision）
        layers = [output] + list(output.get("aux_outputs", []))
        for li, out in enumerate(layers):
            l_cls = self._loss_labels(out, labels)
            l_bce, l_dice = self._loss_masks(out, present, tgt, valid_f)
            layer_loss = (self.cls_weight * l_cls
                          + self.bce_weight * l_bce + self.dice_weight * l_dice)
            total = layer_loss if total is None else total + layer_loss
            if li == 0:
                log = {"cls": float(l_cls.detach()), "bce": float(l_bce.detach()), "dice": float(l_dice.detach())}
        return total, log
