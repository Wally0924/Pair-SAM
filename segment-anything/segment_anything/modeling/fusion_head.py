import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDWConvFusion(nn.Module):
    """
    [Mask2Former-style 後置精修] 重設計版本。

    核心設計改動（相對 v1）：
      1. cross_class_mixer 移至 DW Conv 之前
         原因：空間平滑應作用在「已知各 class 互相競爭結果」的 feature 上，
         而非在各 class 獨立演化的 logit 上。先競爭再平滑，DW Conv 平滑的
         方向才能與正確的類別邊界對齊。
      2. cross_class_mixer 最後一層改用 small-init（std=0.01）而非 zero-init
         原因：zero-init 需要 CE loss 先把輸出從 0 推離，等待期長達數千 iter。
         small-init 讓 mixer 從訓練第一步就有微小輸出，梯度立即流動。
      3. DW Conv 保持 dilation=1+2（有效感受野 ~7×7），做空間平滑。

    結構：
      x → cross_class_mixer（跨類別競爭，residual add）
        → DW Conv dilation=1 → GroupNorm
        → DW Conv dilation=2 → GroupNorm
        → residual add（with original x）
    """

    def __init__(self, num_classes: int = 19):
        super().__init__()

        # Step 1: Cross-class competition
        # 19 → 64 → 19，非線性讓模型學習非線性的抑制關係
        # small-init 最後一層：訓練初期輸出微小但非零，梯度立即流動
        self.cross_class_mixer = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=True),
        )
        nn.init.normal_(self.cross_class_mixer[-1].weight, std=0.01)
        nn.init.zeros_(self.cross_class_mixer[-1].bias)

        # Step 2: Spatial smoothing（在競爭已建立的 feature 上做局部平滑）
        # Per-class instance-style normalization，避免 absent class 填充值污染其他類別
        self.dw_conv1 = nn.Conv2d(
            num_classes, num_classes, kernel_size=3, padding=1, dilation=1,
            groups=num_classes, bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=num_classes, num_channels=num_classes)

        # dilation=2 讓兩層串接後有效感受野 ~7×7
        self.dw_conv2 = nn.Conv2d(
            num_classes, num_classes, kernel_size=3, padding=2, dilation=2,
            groups=num_classes, bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=num_classes, num_channels=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_classes, H, W) — 各類別 logit map
        Returns:
            out: (B, num_classes, H, W) — 精修後的 logit map
        """
        identity = x

        # 1. Cross-class competition（residual add 保護訓練穩定性）
        x = x + self.cross_class_mixer(x)

        # 2. Spatial smoothing on competition-adjusted features
        out = self.norm1(self.dw_conv1(x))
        out = self.norm2(self.dw_conv2(out))

        return identity + out
