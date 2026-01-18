import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d 

class MaskEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,       # 輸入通常是 RGB 顏色的遮罩 (3 channels)
        embed_dim: int = 256,    # 目標輸出維度，必須對齊 SAM 的 Image Encoder (256)
        activation: type[nn.Module] = nn.GELU,
    ):
        """
        Mask Encoder: 將檢索到的語意分割遮罩 (Reference Mask) 編碼為特徵圖。
        
        Args:
            in_chans (int): 輸入遮罩的通道數。如果是彩色遮罩(RGB)則為3；如果是類別索引(Index)則建議先轉為Embedding或One-hot。
                            這裡預設處理 RGB 視覺化後的遮罩。
            embed_dim (int): 輸出的特徵維度 (SAM ViT output dimension)。
        """
        super().__init__()
        
        # 結構設計：逐步下採樣 (Downsampling)
        # Input: (B, 3, 1024, 1024)
        
        # Stage 1: /2 (1024 -> 512)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            LayerNorm2d(64),
            activation(),
        )
        
        # Stage 2: /4 (512 -> 256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(128),
            activation(),
        )
        
        # Stage 3: /8 (256 -> 128)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(256),
            activation(),
        )
        
        # Stage 4: /16 (128 -> 64) -> 這是最終目標尺寸
        # 這裡不改變通道數 (維持 256)，但再次下採樣以匹配 ViT 的輸出
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(embed_dim),
            activation(),
        )
        
        # 額外的 1x1 Conv 用於特徵平滑與混合 (Optional)
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # 初始化權重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Reference Mask, shape (B, 3, H, W) 
                              假設 H, W 為 1024 (與影像輸入相同)
        Returns:
            torch.Tensor: Encoded Mask Features, shape (B, 256, H/16, W/16)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output_proj(x)
        return x


# class ClassIDMaskEncoder(nn.Module):
#     def __init__(self, num_classes=20, embed_dim=256):
#         super().__init__()
#         # 1. Embedding Layer: 將整數類別轉為向量 (B, 1, H, W) -> (B, H, W, 64) -> (B, 64, H, W)
#         self.embedding = nn.Embedding(num_classes, 64) 
        
#         # 接下來接上面的 CNN 結構，但在 conv1 的 in_chans 要設為 64
#         self.encoder_cnn = MaskEncoder(in_chans=64, embed_dim=embed_dim) 

#     def forward(self, x):
#         # x shape: (B, 1, H, W)
#         x = x.squeeze(1).long() # (B, H, W)
#         x = self.embedding(x)   # (B, H, W, 64)
#         x = x.permute(0, 3, 1, 2) # (B, 64, H, W)
#         return self.encoder_cnn(x)