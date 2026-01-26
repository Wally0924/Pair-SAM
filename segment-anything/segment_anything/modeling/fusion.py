import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossViewAlignment(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Module 2: Cross-View Attention
        負責將模糊的當前影像特徵 (Query) 與清晰的記憶遮罩特徵 (Key, Value) 進行對齊。
        
        Args:
            embed_dim (int): 特徵維度，需與 SAM Image Encoder 輸出一致 (Default: 256)。
            num_heads (int): Attention Head 數 (Default: 8)。
        """
        super().__init__()
        
        # 使用 PyTorch 內建的 MultiheadAttention
        # batch_first=True 讓輸入格式為 (Batch, Seq_Len, Dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 使用標準 LayerNorm，因為在 Attention 計算時我們會將特徵攤平成 (B, N, C)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, f_curr: torch.Tensor, f_ref: torch.Tensor, ref_void_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            f_curr (Tensor): 當前影像特徵 (Query), shape (B, C, H, W)
            f_ref (Tensor):  參考遮罩特徵 (Key, Value), shape (B, C, H, W)
            ref_void_mask (Tensor): 參考遮罩的 Void Mask，shape (B, H, W)
        Returns:
            f_align (Tensor): 對齊後的特徵圖, shape (B, C, H, W)
        """
        b, c, h, w = f_curr.shape
        
        # 1. Reshape for Attention: (B, C, H, W) -> (B, H*W, C)
        # Flatten spatial dimensions into sequence
        key_padding_mask = None
        if ref_void_mask is not None:
            # ref_void_mask 原始尺寸是 (B, 1024, 1024)，我們需要縮小到 (B, H, W) 即 (B, 64, 64)
            # 因為 Attention 是在特徵層級運作的
            
            # 轉換為 Float 才能進行 Interpolate
            mask_float = ref_void_mask.unsqueeze(1).float() # (B, 1, 1024, 1024)
            
            # 使用 Nearest Neighbor 插值，確保原本是黑色的地方縮小後還是標記為黑
            # 使用 Max Pooling 也可以，這裡用 interpolate 比較直觀
            mask_downsampled = F.interpolate(
                mask_float, 
                size=(h, w), 
                mode='nearest'
            ) # (B, 1, H, W)
            
            # 攤平成 (B, H*W) 以符合 MultiheadAttention 的要求
            # shape: (B, N)
            key_padding_mask = mask_downsampled.flatten(2).squeeze(1).bool()

        q = f_curr.flatten(2).transpose(1, 2) # [B, N, C], where N = H*W
        k = f_ref.flatten(2).transpose(1, 2)  # [B, N, C]
        v = k                                 # Value 也是來自 Reference
        
        # 2. Cross Attention
        # attn_output: [B, N, C]
        attn_output, _ = self.attn(query=q, key=k, value=v, key_padding_mask=key_padding_mask)
        
        # 3. Residual Connection & Norm
        # 這裡保留了 Residual (q + attn)，確保至少有原始影像的資訊
        f_align = self.norm(q + attn_output)
        
        # 4. Reshape back to spatial: (B, N, C) -> (B, C, H, W)
        f_align = f_align.transpose(1, 2).view(b, c, h, w)
        
        return f_align


class GatedFusion(nn.Module):
    def __init__(self, embed_dim: int = 256):
        """
        Module 2: Gate Fusion
        計算融合權重 alpha，並執行加權融合。
        公式: F_fuse = (1 - alpha) * F_curr + alpha * F_align
        """
        super().__init__()
        
        # 用一個輕量級 CNN 來預測每個像素的 alpha 值
        # 輸入是 F_curr 和 F_align 的串接 (Channels * 2) -> 輸出 1 channel 的 alpha map
        self.gate_net = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
            nn.Sigmoid() # 將輸出限制在 0~1 之間作為權重
        )

    def forward(self, f_curr: torch.Tensor, f_align: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_curr: 原始影像特徵
            f_align: 對齊後的參考特徵 (來自 CrossViewAlignment)
        Returns:
            f_fuse: 融合後的特徵，將送入 Mask Decoder
        """
        # 1. Concatenate along channel dimension
        cat_feat = torch.cat([f_curr, f_align], dim=1) # (B, 512, H, W)
        
        # 2. Predict Alpha map
        alpha = self.gate_net(cat_feat) # (B, 1, H, W)
        
        # 3. Weighted Fusion
        # alpha 越大，代表模型越依賴 "記憶 (Reference)"
        # alpha 越小，代表模型越依賴 "當前視覺 (Current)"
        f_fuse = (1 - alpha) * f_curr + alpha * f_align
        
        return f_fuse