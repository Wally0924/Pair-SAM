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
      x → 3×3 DW Conv dilation=1（局部平滑）
        → 3×3 DW Conv dilation=2（擴大至約 7×7 感受野）
        → 1×1 PW Conv（跨類別線性混合／互斥抑制）
        → residual add
        → 缺席通道恢復為 ABSENT_FILL

    zero-init 設計：pw_conv 權重與 bias 全初始化為 0，
    訓練起始時 F(x) = 0，output = identity，讓模型穩定收斂後再逐漸學習修正量。
    """

    def __init__(self, num_classes: int = 19):
        super().__init__()

        # 3×3 Depthwise Conv：每個 class channel 獨立做局部空間平滑。
        self.dw_conv1 = nn.Conv2d(
            num_classes, num_classes, kernel_size=3, padding=1, dilation=1,
            groups=num_classes, bias=False
        )
        # Per-class instance-style normalization，避免 absent class 的 -10 填充值污染其他類別通道。
        self.norm1 = nn.GroupNorm(num_groups=num_classes, num_channels=num_classes)

        # dilation=2 讓兩層 DWConv 串接後覆蓋約 7×7 鄰域，補強破碎 logits 的局部一致性。
        self.dw_conv2 = nn.Conv2d(
            num_classes, num_classes, kernel_size=3, padding=2, dilation=2,
            groups=num_classes, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups=num_classes, num_channels=num_classes)

        # 1×1 Pointwise Conv：跨類別線性混合，學習類別間的相互抑制關係
        self.pw_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True)

        # zero-init：確保訓練起始時此模組輸出等同 identity
        nn.init.zeros_(self.pw_conv.weight)
        nn.init.zeros_(self.pw_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_classes, H, W) — 各類別 logit map（全 19 類，decoder 直接輸出）
        Returns:
            out: (B, num_classes, H, W) — 精修後的 logit map
        """
        # F(x)：局部平滑 → 中程平滑 → 跨類別混合
        out = self.norm1(self.dw_conv1(x))
        out = self.norm2(self.dw_conv2(out))
        out = self.pw_conv(out)

        # 殘差相加（zero-init 保證初始時 out = x）
        return out + x
