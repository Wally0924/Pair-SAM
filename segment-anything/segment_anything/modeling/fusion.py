import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .common import LayerNorm2d


# =============================================================================
# [DEPRECATED] CrossViewAlignment — 標準 Cross-Attention 版本
#
# 實驗結果（28 epoch）顯示此模組無法自主學習空間對應：
#   - attention entropy ≈ 0.993（所有 head 全部均勻分布）
#   - 等價於對 f_ref 做全局平均，空間先驗完全失效
#
# 根本原因：softmax over N=4096 個位置，梯度量級 ~1/4096，
# 分割 loss 的間接監督不足以驅動 attention 從均勻分布收斂到有意義的對應。
#
# 保留原始程式碼供對照，實際訓練請改用 DeformableCrossViewAlignment。
# =============================================================================
class CrossViewAlignment(nn.Module):
    """
    Cross-attention between degraded (f_curr) and clear (f_ref) ViT-H embeddings.

    Relative Distance Bias (Liu et al., Swin Transformer, ICCV 2021):
        score(i, j) = Q_i · K_j / sqrt(d) + B(Δrow, Δcol)
    B is a learnable (2H-1, 2W-1, num_heads) table indexed by relative offset.
    This gives soft spatial locality preference without enforcing exact alignment,
    which is appropriate for ACDC image pairs with slight viewpoint shift.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        feat_size: int = 64,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.feat_size = feat_size  # H = W = 64

        # Q, K, V projections (replaces nn.MultiheadAttention)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

        # Relative Distance Bias table: shape (2H-1, 2W-1, num_heads)
        # Indexed by (Δrow + H - 1, Δcol + W - 1)
        self.rel_bias = nn.Parameter(
            torch.zeros(2 * feat_size - 1, 2 * feat_size - 1, num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        # Pre-compute relative index table: shape (H*W, H*W) — registered as buffer (not trained)
        self._register_rel_index(feat_size)

        self.norm = LayerNorm2d(embed_dim)
        nn.init.constant_(self.norm.weight, 0.1)
        nn.init.constant_(self.norm.bias, 0.0)

    def _register_rel_index(self, S: int):
        """Pre-compute (N, N) index table into rel_bias for H=W=S."""
        coords = torch.stack(torch.meshgrid(
            torch.arange(S), torch.arange(S), indexing="ij"
        ))  # (2, S, S)
        coords_flat = coords.flatten(1)  # (2, N)

        # Relative offsets: (2, N, N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel[0] += S - 1  # shift to [0, 2S-2]
        rel[1] += S - 1

        # Flatten to single index: row * (2S-1) + col
        rel_index = rel[0] * (2 * S - 1) + rel[1]  # (N, N)
        self.register_buffer("rel_index", rel_index)  # (N, N), long

    def _get_rel_bias(self) -> torch.Tensor:
        """Look up bias for all (i, j) pairs. Returns (num_heads, N, N)."""
        L = (2 * self.feat_size - 1) ** 2
        bias_flat = self.rel_bias.view(L, self.num_heads)  # (L, num_heads)

        N = self.feat_size ** 2
        idx = self.rel_index.view(-1)  # (N*N,)
        bias = bias_flat[idx].view(N, N, self.num_heads)  # (N, N, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, N, N)

    def forward(self, f_curr: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_curr (Tensor): (B, C, H, W) — degraded image features
            f_ref  (Tensor): (B, C, H, W) — clear image features
        Returns:
            f_align (Tensor): (B, C, H, W)
        """
        b, c, h, w = f_curr.shape
        N = h * w

        # 1. Flatten spatial dims: (B, N, C)
        q = f_curr.flatten(2).transpose(1, 2)
        k = f_ref.flatten(2).transpose(1, 2)
        v = f_ref.flatten(2).transpose(1, 2)

        # 2. Project to multi-head Q, K, V
        q = self.q_proj(q).view(b, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(b, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(b, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Attention scores + Relative Distance Bias
        attn = (q @ k.transpose(-2, -1)) / self.scale        # (B, num_heads, N, N)
        attn = attn + self._get_rel_bias().unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 4. Weighted sum of values
        out = (attn @ v).transpose(1, 2).reshape(b, N, c)    # (B, N, C)
        out = self.out_proj(out)

        # 5. Reshape + LayerNorm2d
        f_align = out.transpose(1, 2).view(b, c, h, w)
        f_align = self.norm(f_align)

        return f_align


# =============================================================================
# DeformableCrossViewAlignment — 可形變採樣版本（取代上方的 CrossViewAlignment）
#
# 設計動機：
#   標準 cross-attention 對 N=4096 個位置做 softmax，梯度量級 ~1/4096，
#   分割 loss 的間接監督無法驅動 attention 從均勻分布收斂。
#
#   Deformable attention（Zhu et al., Deformable DETR, ICLR 2021）：
#   每個 query 只預測 K 個採樣點的位置（K=4），softmax 只在 K 個點上做，
#   梯度量級 ~1/K → 比標準版本強 4096/4 = 1024 倍。
#
# 空間先驗初始化：
#   offset_net 初始化為全零，訓練起點為每個 query 採樣自身對應的 f_ref 位置。
#   由分割 loss 驅動偏移量逐漸調整到真正有意義的對應位置。
#
# 監控指標（由 AttentionMonitor 讀取）：
#   _last_offset_mag : float — 偏移量平均絕對值，0=未學習移動，0.5=最大偏移
#   _last_attn_entropy : float [0,1] — K 點上的 normalized entropy，
#                         0=集中在單一採樣點，1=均勻分配到所有 K 點
# =============================================================================
class DeformableCrossViewAlignment(nn.Module):
    """
    Deformable cross-attention between degraded (f_curr) and clear (f_ref) features.

    Each query at spatial position p_i predicts K offset vectors {Δp_k},
    then samples f_ref only at {p_i + Δp_k} via bilinear interpolation.
    Attention weights are applied over K points (softmax over K, not N).

    Reference: Deformable DETR (Zhu et al., ICLR 2021)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,    # K: sampling points per query per head
        feat_size: int = 64,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size

        # Query projection from f_curr
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # Offset network: predicts (num_heads * num_points * 2) offsets per spatial position
        # Init to zeros so training starts with identity sampling (p_i + 0 = p_i)
        self.offset_net = nn.Linear(embed_dim, num_heads * num_points * 2)
        nn.init.zeros_(self.offset_net.weight)
        nn.init.zeros_(self.offset_net.bias)

        # Attention weight network: predicts softmax weights over K points
        self.attn_weight_net = nn.Linear(embed_dim, num_heads * num_points)

        # Value projection from f_ref
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorm2d: gamma init 0.1 to match image_encoder neck output scale
        self.norm = LayerNorm2d(embed_dim)
        nn.init.constant_(self.norm.weight, 0.1)
        nn.init.constant_(self.norm.bias, 0.0)

        # Monitoring attributes (populated during forward, read by AttentionMonitor)
        self._last_offset_mag: float = 0.0
        self._last_attn_entropy: float = 1.0

    def _build_ref_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Build normalized reference grid for identity sampling.
        Returns: (1, H*W, 1, 1, 2) — each query's own position in [-1, 1].
        Note: grid_sample convention uses (x, y) = (col, row).
        """
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1)              # (H, W, 2) — (x, y)
        return grid.view(1, H * W, 1, 1, 2)                       # (1, N, 1, 1, 2)

    def forward(self, f_curr: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_curr : (B, C, H, W) — degraded image features (source of queries)
            f_ref  : (B, C, H, W) — clear reference features (source of values)
        Returns:
            f_align: (B, C, H, W) — aligned features for GatedFusion
        """
        B, C, H, W = f_curr.shape
        N = H * W

        # ── 1. Query projection ───────────────────────────────────────────────
        q_flat = f_curr.flatten(2).transpose(1, 2)   # (B, N, C)
        q = self.q_proj(q_flat)                       # (B, N, C)

        # ── 2. Predict spatial offsets ────────────────────────────────────────
        # Raw offsets: (B, N, num_heads * num_points * 2)
        offsets_raw = self.offset_net(q)
        offsets = offsets_raw.view(B, N, self.num_heads, self.num_points, 2)
        # tanh → [-1, 1], scale by 0.5 → sampling range ±0.5 in normalized coords
        offsets = offsets.tanh() * 0.5

        # ── 3. Sampling locations = identity position + learned offset ────────
        # ref_grid: (1, N, 1, 1, 2) — each query's own position in [-1, 1]
        ref_grid = self._build_ref_grid(H, W, f_curr.device)
        # sampling_locs: (B, N, num_heads, num_points, 2)
        sampling_locs = (ref_grid + offsets).clamp(-1.0, 1.0)

        # ── 4. Value projection from f_ref ────────────────────────────────────
        v_flat = self.v_proj(f_ref.flatten(2).transpose(1, 2))   # (B, N, C)
        v_spatial = v_flat.transpose(1, 2).view(B, C, H, W)      # (B, C, H, W)

        # ── 5. Attention weights (softmax over K points, not N) ───────────────
        # attn_weights: (B, N, num_heads, num_points)
        attn_weights = self.attn_weight_net(q)
        attn_weights = attn_weights.view(B, N, self.num_heads, self.num_points)
        attn_weights = attn_weights.softmax(dim=-1)

        # ── 6. Bilinear sampling + per-head weighted aggregation ──────────────
        head_outputs = []
        for h_idx in range(self.num_heads):
            # Value channels for this head: (B, head_dim, H, W)
            v_h = v_spatial[:, h_idx * self.head_dim:(h_idx + 1) * self.head_dim]

            # Sampling locations for this head: (B, N, num_points, 2)
            locs_h = sampling_locs[:, :, h_idx, :, :]              # (B, N, K, 2)
            locs_h = locs_h.reshape(B, N * self.num_points, 1, 2)  # (B, N*K, 1, 2)

            # grid_sample: (B, head_dim, N*K, 1)
            sampled = F.grid_sample(
                v_h, locs_h,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            # (B, head_dim, N*K) → (B, N*K, head_dim) → (B, N, K, head_dim)
            sampled = sampled.squeeze(-1).permute(0, 2, 1)
            sampled = sampled.view(B, N, self.num_points, self.head_dim)

            # Attention weights for this head: (B, N, K, 1)
            w_h = attn_weights[:, :, h_idx, :].unsqueeze(-1)

            # Weighted sum over K → (B, N, head_dim)
            head_outputs.append((w_h * sampled).sum(dim=2))

        # ── 7. Concat heads + output projection ───────────────────────────────
        out = torch.cat(head_outputs, dim=-1)   # (B, N, C)
        out = self.out_proj(out)

        # ── 8. Reshape + LayerNorm2d ──────────────────────────────────────────
        f_align = out.transpose(1, 2).view(B, C, H, W)
        f_align = self.norm(f_align)

        # ── 9. 更新監控指標（detached，不影響梯度）────────────────────────────
        with torch.no_grad():
            self._last_offset_mag = float(offsets.detach().abs().mean().item())

            # Entropy over K points (normalized by log(K))
            # attn_weights: (B, N, num_heads, K) — already softmax, sum=1
            a = attn_weights.float().mean(dim=[0, 1])       # (num_heads, K)
            ent = -(a * (a + 1e-12).log()).sum(dim=-1)      # (num_heads,)
            max_ent = math.log(self.num_points)
            self._last_attn_entropy = float((ent / max_ent).mean().item())

        return f_align


class GatedFusion(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, f_curr: torch.Tensor, f_align: torch.Tensor) -> torch.Tensor:
        # 1. 預測 Alpha
        cat_feat = torch.cat([f_curr, f_align], dim=1)
        alpha = self.gate_net(cat_feat)

        # 2. 加權融合
        f_fuse = (1 - alpha) * f_curr + alpha * f_align

        # 3. LayerNorm2d
        f_fuse = self.norm(f_fuse)

        return f_fuse
