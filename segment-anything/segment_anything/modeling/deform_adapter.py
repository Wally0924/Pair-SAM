"""WeatherSAM 雙向可變形 Adapter（A3）。SPM → UAWarpC 參考；Injector + Extractor。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops.ms_deform_attn import MSDeformAttn

_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # softplus(x) ≈ 0.05


def get_reference_points(spatial_shapes, device):
    refs = []
    for (H_, W_) in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing='ij')
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        refs.append(torch.stack((ref_x, ref_y), -1))
    reference_points = torch.cat(refs, 1)[:, :, None]  # (1, sum_L, 1, 2)
    return reference_points


def deform_inputs(h, w, device):
    """h,w = ViT token grid（1/16 of input）。value 三尺度 = 1/8,1/16,1/32。"""
    c_shapes = torch.as_tensor([(h * 2, w * 2), (h, w), (h // 2, w // 2)],
                               dtype=torch.long, device=device)
    c_lsi = torch.cat((c_shapes.new_zeros((1,)), c_shapes.prod(1).cumsum(0)[:-1]))
    inject = [get_reference_points([(h, w)], device), c_shapes, c_lsi]

    vit_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=device)
    vit_lsi = torch.cat((vit_shapes.new_zeros((1,)), vit_shapes.prod(1).cumsum(0)[:-1]))
    extract = [get_reference_points([(h * 2, w * 2), (h, w), (h // 2, w // 2)], device),
               vit_shapes, vit_lsi]
    return inject, extract


class ReferencePriorModule(nn.Module):
    """取代 ViT-Adapter SPM：把 UAWarpC 對齊的 VGG 參考轉成 3 尺度 token 流。
    1/8 ← l2；1/16 ← l3；1/32 ← l3 stride-2 降採（決策②）。"""
    def __init__(self, l2_channels=256, l3_channels=512, dim=1280, use_reference=True):
        super().__init__()
        self.dim = dim
        self.use_reference = use_reference
        self.proj_c2 = nn.Conv2d(l2_channels, dim, kernel_size=1)
        self.proj_c3 = nn.Conv2d(l3_channels, dim, kernel_size=1)
        self.down_c4 = nn.Conv2d(l3_channels, dim, kernel_size=3, stride=2, padding=1)
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(self, feats):
        l2 = feats['l2']; l3 = feats['l3']
        B = l2.shape[0]
        c2 = self.proj_c2(l2)                    # (B,dim,H8,W8)
        c3 = self.proj_c3(l3)                    # (B,dim,H16,W16)
        c4 = self.down_c4(l3)                    # (B,dim,H32,W32)

        def _flat(x):
            return x.flatten(2).transpose(1, 2)  # (B, H*W, dim)
        t2, t3, t4 = _flat(c2), _flat(c3), _flat(c4)
        t2 = t2 + self.level_embed[0]
        t3 = t3 + self.level_embed[1]
        t4 = t4 + self.level_embed[2]
        c = torch.cat([t2, t3, t4], dim=1)       # (B, L, dim)

        if not self.use_reference:
            c = torch.zeros_like(c)

        mask = feats.get('mask', None)
        if mask is not None:
            m2 = F.adaptive_avg_pool2d(mask, c2.shape[-2:])
            m3 = F.adaptive_avg_pool2d(mask, c3.shape[-2:])
            m4 = F.adaptive_avg_pool2d(mask, c4.shape[-2:])
            conf = torch.cat([_flat(m2), _flat(m3), _flat(m4)], dim=1)  # (B,L,1)
        else:
            conf = torch.ones(B, c.shape[1], 1, device=c.device, dtype=c.dtype)
        return c, conf
