import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class ActiveBoundaryLoss(nn.Module):
    """
    Active Boundary Loss (ABL)：透過 KL 散度自動偵測預測邊界 (PDB)，
    並利用距離加權的方向性 Cross Entropy 引導 PDB 像素對齊真實邊界 (GTB)。

    核心流程：
        Phase I:  利用 8 鄰域 KL 散度找出預測邊界 (PDB)
        Phase II: 從 PDB 到最近 GTB 的方向向量 → Label Smoothed 的 8 維目標分佈
        Loss:     距離加權的 Soft Cross Entropy
    """

    # 8 個方向的位移量 (dy, dx)，順序固定作為方向索引 0~7
    SHIFTS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, pdb_ratio: float = 0.01, kl_threshold: float = 0.1,
                 smooth_main: float = 0.8, smooth_other: float = 0.2):
        """
        Args:
            pdb_ratio:    PDB 像素佔全圖的最大比例 (預設 1%)。
            kl_threshold: 固定閾值，用於粗篩候選 PDB 像素。
            smooth_main:  Label Smoothing 主方向機率 (預設 0.8)。
            smooth_other: Label Smoothing 其餘 7 方向的機率總和 (預設 0.2)。
        """
        super().__init__()
        self.pdb_ratio = pdb_ratio
        self.kl_threshold = kl_threshold
        self.smooth_main = smooth_main
        self.smooth_other = smooth_other
        self.num_dirs = 8

    def forward(self, fused_logits_hr: torch.Tensor,
                gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            fused_logits_hr: (B, C, H, W) ContextFusionHead 輸出的高解析度 logits。
            gt_mask:         (B, H, W)     Ground Truth (0~18 類別，255 為忽略區)。
        Returns:
            loss: (scalar) Active Boundary Loss。若無有效 PDB 像素則回傳 0。
        """
        device = fused_logits_hr.device
        B, C, H, W = fused_logits_hr.shape

        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        for b in range(B):
            logits_b = fused_logits_hr[b].unsqueeze(0)  # (1, C, H, W)
            gt_b = gt_mask[b]                             # (H, W)

            # --- Phase I: 計算 8 方向 KL 散度，找到 PDB ---
            kl_map = self._compute_kl_map(logits_b)       # (H, W)

            # 建立有效區域遮罩：排除 ignore_index=255
            valid_region = (gt_b != 255)                   # (H, W)
            kl_map = kl_map * valid_region.float()

            # 自適應閾值：固定閾值粗篩 + top-k 硬截斷
            pdb_mask = self._select_pdb(kl_map, H, W)     # (H, W) bool

            # 排除 ignore 區域
            pdb_mask = pdb_mask & valid_region

            num_pdb = pdb_mask.sum().item()
            if num_pdb == 0:
                continue

            # --- Phase II: 計算 GTB 距離映射與方向目標 ---
            gt_boundary, dist_map = self._compute_gt_boundary_and_dist(
                gt_b, device
            )  # 皆為 (H, W) tensor

            # 取出 PDB 像素的座標
            pdb_coords = torch.nonzero(pdb_mask, as_tuple=False)  # (N, 2) [y, x]
            num_pdb = pdb_coords.shape[0]

            # 計算每個 PDB 像素到最近 GTB 的方向 → 8 維 Label Smoothed 目標分佈
            D_g = self._compute_direction_target(
                pdb_coords, dist_map, device
            )  # (N, 8)

            # --- 取出 PDB 像素的 8 方向 KL 值作為預測方向分佈的 logits ---
            direction_logits = self._get_direction_logits_at_pdb(
                logits_b, pdb_coords
            )  # (N, 8)

            # --- 距離加權的 Soft Cross Entropy ---
            # 手動 Soft CE：F.cross_entropy 不支援 soft label
            log_D_p = F.log_softmax(direction_logits, dim=1)  # (N, 8)
            ce_per_pixel = -(D_g * log_D_p).sum(dim=1)        # (N,)

            # 距離加權：離 GTB 越遠的 PDB 像素，權重越大（鼓勵積極移動）
            dist_weights = dist_map[pdb_coords[:, 0], pdb_coords[:, 1]]
            # 正規化權重避免數值爆炸
            # dist_weights = dist_weights / (dist_weights.max() + 1e-6)
            dist_weights = (dist_weights / 20.0).clamp(max=1.0)

            weighted_loss = (ce_per_pixel * dist_weights).mean()
            total_loss = total_loss + weighted_loss
            valid_samples += 1

        if valid_samples > 0:
            total_loss = total_loss / float(valid_samples)

        return total_loss

    def _compute_kl_map(self, logits: torch.Tensor) -> torch.Tensor:
        """
        計算 8 鄰域 KL 散度的加總映射圖。

        使用 torch.roll 向量化平移（非逐像素迴圈），
        並在每個方向遮罩掉因循環回繞產生的無效邊緣。

        Args:
            logits: (1, C, H, W) 單張影像的 logits。
        Returns:
            kl_total: (H, W) 8 方向 KL 散度的加總值。
        """
        H, W = logits.shape[2], logits.shape[3]
        kl_total = torch.zeros(H, W, device=logits.device)

        # 中央像素的 log-probability（保留梯度）
        log_p = F.log_softmax(logits, dim=1)  # (1, C, H, W)

        for dy, dx in self.SHIFTS:
            # 平移鄰近像素（.detach() → Conflict Suppression 核心）
            shifted = torch.roll(logits, shifts=(dy, dx), dims=(2, 3))
            q = F.softmax(shifted.detach(), dim=1)  # (1, C, H, W)

            # KL(P || Q) 逐像素
            kl_dir = F.kl_div(
                log_p, q, reduction='none', log_target=False
            ).sum(dim=1).squeeze(0)  # (H, W)

            # ★ 修正 wrap-around：遮罩掉因循環回繞產生的無效邊緣
            if dy == -1:
                kl_dir[0, :] = 0
            if dy == 1:
                kl_dir[-1, :] = 0
            if dx == -1:
                kl_dir[:, 0] = 0
            if dx == 1:
                kl_dir[:, -1] = 0

            kl_total = kl_total + kl_dir

        return kl_total

    def _select_pdb(self, kl_map: torch.Tensor,
                    H: int, W: int) -> torch.Tensor:
        """
        兩階段自適應閾值選取 PDB 像素：
        Step 1: 固定閾值粗篩
        Step 2: top-k 硬截斷，確保 PDB < 全圖 pdb_ratio

        Returns:
            pdb_mask: (H, W) bool tensor
        """
        max_pdb_pixels = int(H * W * self.pdb_ratio)

        # Step 1: 固定閾值粗篩
        candidates = kl_map > self.kl_threshold

        num_candidates = candidates.sum().item()

        if num_candidates == 0:
            return candidates  # 沒有候選像素

        if num_candidates <= max_pdb_pixels:
            return candidates  # 候選數已在限制內

        # Step 2: top-k 硬截斷
        candidate_values = kl_map[candidates]
        topk_values, _ = torch.topk(candidate_values, max_pdb_pixels)
        adaptive_threshold = topk_values[-1]
        pdb_mask = kl_map >= adaptive_threshold

        return pdb_mask

    def _compute_gt_boundary_and_dist(
        self, gt_mask: torch.Tensor, device: torch.device
    ) -> tuple:
        """
        從 Ground Truth Mask 提取邊界 (GTB) 與距離映射圖 (EDT)。

        ★ 在計算 EDT 之前過濾 ignore_index=255，
          避免未標註區域被誤判為物件邊界。

        Args:
            gt_mask: (H, W) Ground Truth (0~18, 255=ignore)
            device:  目標 GPU device
        Returns:
            gt_boundary: (H, W) bool tensor，True = 邊界像素
            dist_map:    (H, W) float tensor，每個像素到最近 GTB 的距離
        """
        gt_np = gt_mask.cpu().numpy().astype(np.int32)

        # 將 ignore 區域替換為 0（背景），避免污染邊界提取
        gt_clean = gt_np.copy()
        gt_clean[gt_np == 255] = 0

        # 邊界提取：與任一 8 鄰域類別不同的像素即為邊界
        boundary = np.zeros_like(gt_clean, dtype=bool)
        for dy, dx in self.SHIFTS:
            shifted = np.roll(np.roll(gt_clean, dy, axis=0), dx, axis=1)
            boundary = boundary | (gt_clean != shifted)

        # 邊緣行列修正（np.roll 同樣有 wrap-around 問題）
        boundary[0, :] = False
        boundary[-1, :] = False
        boundary[:, 0] = False
        boundary[:, -1] = False

        # 排除 ignore 區域的邊界
        boundary[gt_np == 255] = False

        # EDT：計算每個「非邊界」像素到最近邊界的距離
        # distance_transform_edt 計算的是 True 像素到最近 False 的距離
        # 所以我們取反：非邊界像素到最近邊界的距離
        if boundary.sum() == 0:
            # 若無有效邊界，距離全為 0
            dist_np = np.zeros_like(gt_clean, dtype=np.float32)
        else:
            dist_np = distance_transform_edt(~boundary).astype(np.float32)

        gt_boundary = torch.from_numpy(boundary).to(device)
        dist_map = torch.from_numpy(dist_np).to(device)

        return gt_boundary, dist_map

    def _compute_direction_target(
        self, pdb_coords: torch.Tensor,
        dist_map: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        依照 ABL 論文 Eq.2，對每個 PDB 像素在 8 個鄰域方向中，
        取 EDT 距離值最小的方向作為 ground-truth 方向目標（argmin）。

        D_g_i = Φ(argmin_j M_{i+Δj}),  j ∈ {0,...,7}

        直接查詢 8 個鄰居的 EDT 值取 argmin，嚴格符合論文定義，
        避免平坦 EDT 區域 np.gradient 估計不穩定的問題。

        Args:
            pdb_coords: (N, 2) PDB 像素座標 [y, x]
            dist_map:   (H, W) float tensor，EDT 距離映射圖
            device:     GPU device
        Returns:
            D_g: (N, 8) label-smoothed soft target distribution
        """
        N = pdb_coords.shape[0]
        dist_np = dist_map.cpu().numpy()
        H, W = dist_np.shape

        pdb_y = pdb_coords[:, 0].cpu().numpy()
        pdb_x = pdb_coords[:, 1].cpu().numpy()

        # 對 8 個鄰域方向逐一查詢 EDT 值 → (N, 8)
        neighbor_dists = np.full((N, 8), np.inf, dtype=np.float32)
        for d_idx, (dy, dx) in enumerate(self.SHIFTS):
            ny = pdb_y + dy
            nx = pdb_x + dx
            valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            neighbor_dists[valid, d_idx] = dist_np[ny[valid], nx[valid]]

        # argmin：選 EDT 距離最小（最靠近 GTB）的鄰域方向
        main_direction_idx = torch.from_numpy(
            np.argmin(neighbor_dists, axis=1)
        ).long().to(device)  # (N,)

        # Label Smoothing：主方向 0.8，其餘均分 0.2/7
        other_prob = self.smooth_other / (self.num_dirs - 1)
        D_g = torch.full((N, self.num_dirs), other_prob, device=device)
        D_g[torch.arange(N, device=device), main_direction_idx] = self.smooth_main

        return D_g

    def _get_direction_logits_at_pdb(
        self, logits: torch.Tensor,
        pdb_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        取出 PDB 像素與 8 個鄰居之間的 KL 散度值，作為方向預測的 logits。

        Args:
            logits:     (1, C, H, W) 單張影像的 logits。
            pdb_coords: (N, 2) PDB 像素座標 [y, x]
        Returns:
            direction_logits: (N, 8) 每個 PDB 像素在 8 個方向上的 KL 散度值
        """
        N = pdb_coords.shape[0]
        H, W = logits.shape[2], logits.shape[3]
        device = logits.device

        log_p = F.log_softmax(logits, dim=1)  # (1, C, H, W)
        direction_logits = torch.zeros(N, self.num_dirs, device=device)

        for d_idx, (dy, dx) in enumerate(self.SHIFTS):
            shifted = torch.roll(logits, shifts=(dy, dx), dims=(2, 3))
            q = F.softmax(shifted.detach(), dim=1)

            kl_full = F.kl_div(
                log_p, q, reduction='none', log_target=False
            ).sum(dim=1).squeeze(0)  # (H, W)

            # 修正 wrap-around 邊緣
            if dy == -1:
                kl_full[0, :] = 0
            if dy == 1:
                kl_full[-1, :] = 0
            if dx == -1:
                kl_full[:, 0] = 0
            if dx == 1:
                kl_full[:, -1] = 0

            # 取出 PDB 像素對應的 KL 值
            direction_logits[:, d_idx] = kl_full[pdb_coords[:, 0], pdb_coords[:, 1]]

        return direction_logits


class ContextLoss(nn.Module):
    """
    計算 ResidualDWConvFusion 輸出的 19 類別高解析度 logits 的 Cross Entropy Loss。
    加入 Class-Balanced Weights，強迫模型重視被吃掉的小物件類別。
    """
    def __init__(self, ce_weight: float = 1.0, num_classes: int = 19):
        super().__init__()
        self.ce_weight = ce_weight

        # Cityscapes 類別倒數頻率權重（CLASS_MAP 順序）
        # road=0, sidewalk=1, building=2, wall=3, fence=4, pole=5,
        # traffic light=6, traffic sign=7, vegetation=8, terrain=9,
        # sky=10, person=11, rider=12, car=13, truck=14, bus=15,
        # train=16, motorcycle=17, bicycle=18
        weights = torch.ones(num_classes)

        # 小物件/稀有類別 → 加重懲罰（強迫模型不忽略它們）
        # wall, fence, pole, traffic light, traffic sign,
        # person, rider, truck, bus, train, motorcycle, bicycle
        heavy_classes = [3, 4, 5, 6, 7, 11, 12, 14, 15, 16, 17, 18]
        for c in heavy_classes:
            weights[c] = 5.0

        # register_buffer 確保 weights 跟模型一起搬到 GPU
        self.register_buffer('class_weights', weights)
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=255
        )

    def forward(self, fused_logits_hr: torch.Tensor, gt_mask: torch.Tensor):
        """
        Inputs:
            fused_logits_hr: (B, 19, H, W) 融合後的 19 類別高解析度特徵圖。
            gt_mask: (B, H, W) 標註的 Ground Truth (0~18 類別，255 為忽略區)。
        Returns:
            total_loss: (scalar) 加權後的 Cross Entropy Loss。
            ce_val: (float) 原始未加權的 Cross Entropy Loss 數值。
        """
        # Guard：若 batch 中所有像素都是 ignore_index=255，weighted mean 分母為 0 → NaN
        # 此時回傳 0 loss（視為此 sample 對 CE 無貢獻），避免污染整個訓練 average
        valid_pixel_count = (gt_mask != 255).sum()
        if valid_pixel_count == 0:
            zero = torch.zeros((), device=fused_logits_hr.device, dtype=fused_logits_hr.dtype)
            return zero, 0.0

        ce_loss = self.ce_loss_fn(fused_logits_hr, gt_mask)
        total_loss = self.ce_weight * ce_loss
        return total_loss, total_loss.item()



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


