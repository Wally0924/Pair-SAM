# ============================================================================
# Vendored from: facebookresearch/detectron2
#   File: detectron2/modeling/backbone/vit.py, class SimpleFeaturePyramid
#   Commit: 02b5c4e295e990042a714712c21dc79b731e8833
#   License: Apache-2.0 (Copyright (c) Facebook, Inc. and its affiliates.)
#   Paper: Li et al., "Exploring Plain Vision Transformer Backbones for
#          Object Detection" (ViTDet), ECCV 2022. arXiv:2203.16527
#
# [WeatherSAM adaptations]（完整清單）:
#   1. 移除 detectron2 Backbone 基類 / ShapeSpec / top_block(p6)，改為純 nn.Module。
#   2. detectron2 的 Conv2d+get_norm("LN") wrapper → nn.Conv2d + 本 repo LayerNorm2d
#      （兩者等價：channel-wise LayerNorm over (C,H,W)）。
#   3. 輸出從 {"p2".."p5"} dict 改為 ([1/32, 1/16, 1/8], mask_features_1/4)，
#      對齊 Mask2Former transformer decoder 的 coarse→fine round-robin 餵入序。
#   4. 輸入為 SAM neck 之後的 256-d 特徵（上游為 ViT 主幹 768/1024-d 輸出）。
# ============================================================================
import torch
from torch import nn

from .common import LayerNorm2d


class SimpleFPN(nn.Module):
    """ViTDet Simple Feature Pyramid（見檔頭出處）。"""

    def __init__(self, dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.stages = nn.ModuleList()
        # 上游 SimpleFeaturePyramid：scale_factors (4.0, 2.0, 1.0, 0.5)
        for scale in (4.0, 2.0, 1.0, 0.5):
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_ch = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_ch = dim // 2
            elif scale == 1.0:
                layers = []
                out_ch = dim
            else:  # 0.5
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                out_ch = dim
            # 上游：每尺度接 1x1 conv(bias=False)+LN → 3x3 conv(bias=False)+LN
            layers.extend([
                nn.Conv2d(out_ch, out_dim, kernel_size=1, bias=False),
                LayerNorm2d(out_dim),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_dim),
            ])
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor):
        """x: (B, 256, 64, 64) — SAM neck 輸出（stride 16）。"""
        f4 = self.stages[0](x)    # (B, 256, 256, 256)  1/4 → mask features
        f8 = self.stages[1](x)    # (B, 256, 128, 128)  1/8
        f16 = self.stages[2](x)   # (B, 256, 64, 64)    1/16
        f32 = self.stages[3](x)   # (B, 256, 32, 32)    1/32
        # [WeatherSAM adaptation] coarse→fine list + 獨立 mask features
        return [f32, f16, f8], f4
