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
