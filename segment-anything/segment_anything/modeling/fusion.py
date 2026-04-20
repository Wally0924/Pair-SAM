import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .common import LayerNorm2d


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
        # 初始化 gamma=0.1，使 f_align 輸出 norm 與 image_encoder neck LayerNorm2d 的輸出尺度一致
        # image_encoder neck 訓練後 gamma ≈ 0.1（norm ~1.68 vs 原始 √256 ≈ 16）
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
        # rel_bias: (2S-1, 2S-1, num_heads) → flatten spatial → (L, num_heads)
        L = (2 * self.feat_size - 1) ** 2
        bias_flat = self.rel_bias.view(L, self.num_heads)  # (L, num_heads)

        # rel_index: (N, N) — index into bias_flat
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
        q = self.q_proj(q).view(b, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = self.k_proj(k).view(b, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(b, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Attention scores + Relative Distance Bias
        # score(i,j) = Q_i · K_j / sqrt(d) + B(Δrow, Δcol)
        attn = (q @ k.transpose(-2, -1)) / self.scale        # (B, num_heads, N, N)
        attn = attn + self._get_rel_bias().unsqueeze(0)       # broadcast over B
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 4. Weighted sum of values
        out = (attn @ v).transpose(1, 2).reshape(b, N, c)    # (B, N, C)
        out = self.out_proj(out)

        # 5. Reshape to spatial + LayerNorm2d（無 residual）
        # 移除 f_curr residual：讓輸出純粹代表從 f_ref cross-attention 提取的對齊特徵。
        # f_curr 與 f_align 的混合比例由下游 GatedFusion 的 alpha gate 決定，
        # 避免 f_curr 在此處與 GatedFusion 被雙重疊加而壓制 reference 訊號。
        # 使用 LayerNorm2d 與 image_encoder neck 保持一致的正規化方式（跨 channel，空間感知）。
        f_align = out.transpose(1, 2).view(b, c, h, w)  # (B, C, H, W)
        f_align = self.norm(f_align)

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

        # 3. LayerNorm2d（與 CrossViewAlignment 及 image_encoder neck 格式一致）
        f_fuse = self.norm(f_fuse)

        return f_fuse
