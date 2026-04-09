import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextFusionHead(nn.Module):
    """
    輕量級的特徵融合頭，接收 N 個獨立的類別預測結果 (例如由 argmax 挑選出最好的 Mask)。
    核心機制包含兩個階段：
    1. 空間平滑 (Spatial Smoothing)：先用 3x3 Depthwise Conv 修補破洞，保留細長結構。
    2. 空間感知通道注意力 (Spatial-Aware Channel Attention)：用 4x4 spatial pool 保留
       區域空間資訊，避免全圖平均丟失位置線索，強制各類別間的互斥性。

    修正重點（相較於舊版）：
    - 移除 LayerNorm：避免 -10.0 填充通道拉偏 norm 統計量，改為對有效通道做
      per-channel InstanceNorm，缺席類別的通道不參與正規化。
    - 調整流程順序：先空間平滑再做 channel attention，讓 attention 作用在形狀
      已修補完整的特徵上，語意更清晰。
    - 改用 4x4 spatial pool：保留一定程度的空間分佈，讓不同區域的類別競爭可以分開處理。
    - 加入可學習的 residual_scale：初始值 0.1，讓訓練初期以 identity 為主，
      隨模型成熟再逐漸增大 fusion 的影響力，避免 out 與 identity 值域不對稱。
    """

    ABSENT_FILL = -10.0  # trainer 填入缺席類別的常數，需與 weather_trainer.py 保持一致

    def __init__(self, num_classes: int = 19, hidden_dim: int = 64):
        super().__init__()

        self.num_classes = num_classes

        # --- 0. Per-channel norm，只正規化有效通道 ---
        # 使用 InstanceNorm2d (affine=True)，逐通道正規化，不受缺席通道 -10.0 的干擾
        self.input_norm = nn.InstanceNorm2d(num_classes, affine=True, eps=1e-5)

        # --- 1. Spatial Smoothing（先做，讓 attention 看到形狀完整的特徵）---
        self.conv1 = nn.Conv2d(num_classes, hidden_dim, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 Depthwise Conv：保留細節邊緣，避免 7x7 過度平滑吃掉細長結構
        self.depthwise_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)

        # 升回 num_classes 供 attention 使用
        self.proj_back = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=False)

        # --- 2. Spatial-Aware Channel Attention ---
        # 4x4 pool 保留空間分佈（不同區域的類別競爭分開處理）
        self.spatial_pool = nn.AdaptiveAvgPool2d(4)

        # attention 在 4x4 spatial 上操作，輸出仍是 (B, C, 4, 4)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # --- 3. 最終分類頭 ---
        self.classifier = nn.Conv2d(num_classes, num_classes, kernel_size=1)

        # --- 4. 可學習的 residual scale ---
        # 初始化為 0.1：訓練初期以 identity 為主，讓 fusion head 穩定學習後再增大影響
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: (B, num_classes, H, W) 各類別獨立預測結果。
               未出現的類別通道填入 ABSENT_FILL (-10.0)。
        Returns:
            out: (B, num_classes, H, W) 融合後的預測結果。
        """
        b, c, h, w = x.shape

        # --- 0. 建立有效通道遮罩，缺席通道不參與 norm ---
        # absent_mask: (B, C, 1, 1)，True 表示該通道為缺席填充
        absent_mask = (x <= self.ABSENT_FILL + 1.0).all(dim=-1).all(dim=-1).unsqueeze(-1).unsqueeze(-1)

        # InstanceNorm2d 逐通道正規化，天然不受其他通道影響
        x_normed = self.input_norm(x)

        # 缺席通道正規化後恢復為固定負值，避免 norm 把 -10.0 變成奇怪的數值
        x_normed = torch.where(absent_mask.expand_as(x_normed),
                               torch.full_like(x_normed, self.ABSENT_FILL),
                               x_normed)

        identity = x_normed  # 殘差分支

        # --- 1. 空間平滑 ---
        feat = self.relu(self.gn1(self.conv1(x_normed)))
        feat = self.relu(self.gn2(self.depthwise_conv(feat)))
        feat = self.proj_back(feat)  # (B, num_classes, H, W)

        # --- 2. Spatial-Aware Channel Attention ---
        # pool 到 4x4，保留空間分佈
        spatial_desc = self.spatial_pool(feat)           # (B, C, 4, 4)
        attn = self.channel_attention(spatial_desc)      # (B, C, 4, 4)

        # 雙線性插值回原始解析度，廣播乘回特徵
        attn = F.interpolate(attn, size=(h, w), mode='bilinear', align_corners=False)
        feat = feat * attn                               # (B, C, H, W)

        # 缺席通道的 attention 結果也強制清零，避免插值產生的殘留值
        feat = torch.where(absent_mask.expand_as(feat),
                           torch.zeros_like(feat),
                           feat)

        # --- 3. 分類頭 ---
        out = self.classifier(feat)                      # (B, num_classes, H, W)

        # --- 4. 殘差：可學習 scale 控制 fusion 影響力 ---
        return out * self.residual_scale + identity
