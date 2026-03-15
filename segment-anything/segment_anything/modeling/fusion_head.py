import torch
import torch.nn as nn

class ContextFusionHead(nn.Module):
    """
    輕量級的特徵融合頭，接收 N 個獨立的類別預測結果 (例如由 argmax 挑選出最好的 Mask)。
    核心機制包含兩個階段：
    1. 全域上下文注意力 (Global Context Attention)：強制各類別間的互斥性 (Mutual Exclusivity)。
    2. 空間平滑 (Spatial Smoothing)：使用大核深度可分離卷積 (7x7 Depthwise Conv) 來修補破洞 (例如反光)。
    """
    def __init__(self, num_classes: int = 19, hidden_dim: int = 64):
        super().__init__()
        
        # --- 0. Pre-Norm: 正規化輸入 logits 的類別維度，避免極端值導致 float16 溢出 ---
        self.input_norm = nn.LayerNorm(num_classes)
        
        # --- 1. Global Context Attention (Mutual Exclusivity) ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 建立通道注意力機制 (類似 SE-Net/CBAM)，用於學習全域類別關係，互相壓抑或增強
        self.channel_attention = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # --- 2. Spatial Context (Hole Patching) ---
        self.conv1 = nn.Conv2d(num_classes, hidden_dim, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # 使用 7x7 Depthwise Conv 擴充感受野，平滑邊界，且不會大幅增加參數量
        self.depthwise_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: (B, num_classes, H, W) 經過挑選後的各類別獨立預測結果。
               未出現的類別通道通常會填入負常數 (例如 -10.0)。
        Returns:
            out: (B, num_classes, H, W) 融合並經過互相排除、平滑化後的預測結果。
        """
        # --- Pre-Norm: 正規化後再作為殘差 identity ---
        b, c, h, w = x.shape
        x = self.input_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, C, H, W)
        identity = x
        
        # --- 階段 1：全域上下文注意力 ---
        # 壓縮出全域特徵: (B, C, H, W) -> (B, C, 1, 1)
        global_desc = self.global_pool(x)
        # 計算通道權重: (B, C, 1, 1)
        attn_weights = self.channel_attention(global_desc)
        # 廣播權重乘回原特徵
        x = x * attn_weights
        
        # --- 階段 2：大核空間平滑 ---
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.depthwise_conv(out)))
        out = self.classifier(out)
        
        # 加上殘差連接有助於梯度順利流動回未被注意力強化的通道
        return out + identity
