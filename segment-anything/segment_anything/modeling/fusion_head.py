import torch
import torch.nn as nn


class ResidualDWConvFusion(nn.Module):
    """
    [Mask2Former-style 後置精修] 取代 ContextFusionHead 的輕量殘差融合模組。

    設計原則：
      - Mask2Former self-attention 已處理 token 空間的類別競爭
      - 此模組只補足 pixel 空間的局部空間一致性（self-attention 做不到的事）
      - 殘差設計 + zero-init 最終層 → 初始時等同 identity，訓練穩定不爆梯度

    結構：
      x → 3×3 DW Conv（每類別獨立空間平滑）
        → 1×1 PW Conv（跨類別線性混合／互斥抑制）
        → GroupNorm
        → residual add
        → 缺席通道恢復為 ABSENT_FILL

    zero-init 設計：pw_conv 權重與 bias 全初始化為 0，
    訓練起始時 F(x) = 0，output = identity，讓模型穩定收斂後再逐漸學習修正量。
    """

    ABSENT_FILL = -10.0  # 與 weather_trainer.py 缺席類別填充值保持一致

    def __init__(self, num_classes: int = 19):
        super().__init__()

        # 3×3 Depthwise Conv：每個 class channel 獨立做局部空間平滑
        # groups=num_classes → 不做跨類別混合，純空間操作
        self.dw_conv = nn.Conv2d(
            num_classes, num_classes, kernel_size=3, padding=1,
            groups=num_classes, bias=False
        )

        # 1×1 Pointwise Conv：跨類別線性混合，學習類別間的相互抑制關係
        self.pw_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True)

        # GroupNorm（groups=1 等同 per-channel LayerNorm over spatial dims）
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_classes)

        # zero-init：確保訓練起始時此模組輸出等同 identity
        nn.init.zeros_(self.pw_conv.weight)
        nn.init.zeros_(self.pw_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_classes, H, W) — 各類別 logit map，缺席類別填 ABSENT_FILL
        Returns:
            out: (B, num_classes, H, W) — 精修後的 logit map
        """
        # 記錄缺席通道位置：該通道所有像素都 <= ABSENT_FILL + 1.0
        absent_mask = (
            (x <= self.ABSENT_FILL + 1.0)
            .all(dim=-1).all(dim=-1)       # (B, C)
            .unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        )

        # F(x)：空間平滑 → 跨類別混合 → norm
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        out = self.norm(out)

        # 殘差相加（zero-init 保證初始時 out = x）
        out = out + x

        # 缺席通道恢復填充值，避免 conv 污染未出現的類別
        out = torch.where(
            absent_mask.expand_as(out),
            torch.full_like(out, self.ABSENT_FILL),
            out
        )
        return out

