import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossViewAlignment(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Norm 是防止特徵被 0 值稀釋的關鍵
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, f_curr: torch.Tensor, f_ref: torch.Tensor, ref_void_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            f_curr (Tensor): (B, C, H, W)
            f_ref (Tensor):  (B, C, H, W)
            ref_void_mask:   忽略此輸入，因為您決定將 Void 視為特徵
        """
        b, c, h, w = f_curr.shape
        
        # 1. 直接展平，不計算 mask (節省運算)
        q = f_curr.flatten(2).transpose(1, 2) # [B, N, C]
        k = f_ref.flatten(2).transpose(1, 2)  # [B, N, C]
        v = k                                 
        
        # 2. Cross Attention (不使用 mask，讓模型看見黑色區域)
        attn_output, _ = self.attn(query=q, key=k, value=v, key_padding_mask=None)
        
        # 3. Residual Connection & Norm
        f_align = self.norm(q + attn_output)
        
        # 4. Reshape
        f_align = f_align.transpose(1, 2).view(b, c, h, w)
        
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
        # [關鍵修正] 輸出的歸一化層
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_curr: torch.Tensor, f_align: torch.Tensor, ref_void_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. 預測 Alpha
        cat_feat = torch.cat([f_curr, f_align], dim=1) 
        alpha = self.gate_net(cat_feat) 
        
        # 2. 加權融合
        f_fuse = (1 - alpha) * f_curr + alpha * f_align

        # 3. [關鍵修正] LayerNorm 救回被稀釋的特徵
        b, c, h, w = f_fuse.shape
        f_fuse = f_fuse.permute(0, 2, 3, 1) # (B, H, W, C)
        f_fuse = self.norm(f_fuse)
        f_fuse = f_fuse.permute(0, 3, 1, 2) # (B, C, H, W)
        
        return f_fuse