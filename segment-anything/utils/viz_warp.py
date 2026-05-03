"""
Warp 對齊可視化工具
-------------------
將 UAWarpC 計算出的 flow field 應用到原始 RGB 影像上，
與惡劣天氣影像並排儲存，直觀確認對齊品質。

儲存格式（每次呼叫存一張）：
  debug_viz/warp/
    step_{N:06d}_b{b}.png   ← 6 格並排圖
      [惡劣天氣] [晴天原圖] [晴天 warp 後] [差異圖] [confidence] [flow 向量場]
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: [3, H, W], float, 值域 [0, 255] 或已正規化
    回傳 [H, W, 3] uint8 numpy array
    """
    t = img_tensor.detach().float().cpu()
    # 如果是正規化影像（值域在 -3~3），先反正規化
    if t.min() < -0.1:
        t = t * _IMAGENET_STD + _IMAGENET_MEAN
        t = (t * 255).clamp(0, 255)
    elif t.max() <= 1.1:
        t = (t * 255).clamp(0, 255)
    else:
        t = t.clamp(0, 255)
    return t.permute(1, 2, 0).numpy().astype(np.uint8)


def _warp_image(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    用 flow 對原始影像（1024×1024）做 warp。
    flow 的值域為 64×64 feat_px，需要先 upsample + 重新縮放到影像像素單位。

    img:  [3, H_img, W_img]  (1024×1024)
    flow: [2, H_f,   W_f  ]  (64×64，值域 = feat_px)
    """
    H_img, W_img = img.shape[-2:]
    H_f,   W_f   = flow.shape[-2:]

    # flow 值放大到影像像素單位，再 upsample 到影像解析度
    scale_x = W_img / W_f
    scale_y = H_img / H_f
    flow_img = flow.clone().float()
    flow_img[0] *= scale_x
    flow_img[1] *= scale_y
    flow_img = F.interpolate(
        flow_img.unsqueeze(0), size=(H_img, W_img),
        mode='bilinear', align_corners=False
    ).squeeze(0)

    # grid_sample
    xx = torch.arange(W_img, dtype=flow_img.dtype, device=flow_img.device).view(1, -1).expand(H_img, -1)
    yy = torch.arange(H_img, dtype=flow_img.dtype, device=flow_img.device).view(-1, 1).expand(-1, W_img)
    grid = torch.stack([xx, yy], dim=0)          # [2, H, W]
    vgrid = grid + flow_img                        # [2, H, W]
    vgrid[0] = 2.0 * vgrid[0] / max(W_img - 1, 1) - 1.0
    vgrid[1] = 2.0 * vgrid[1] / max(H_img - 1, 1) - 1.0
    vgrid = vgrid.permute(1, 2, 0).unsqueeze(0)   # [1, H, W, 2]

    warped = F.grid_sample(
        img.unsqueeze(0).float(), vgrid,
        mode='bilinear', align_corners=True, padding_mode='zeros'
    ).squeeze(0)                                    # [3, H, W]
    return warped


def save_warp_comparison(
    img_curr:   torch.Tensor,   # [3, H, W] or [B, 3, H, W]，值域 0-255 或 ImageNet 正規化
    img_ref:    torch.Tensor,   # [3, H, W] or [B, 3, H, W]
    flow:       torch.Tensor,   # [2, Hf, Wf] or [B, 2, Hf, Wf]，feat_px 單位
    confidence: torch.Tensor,   # [1, Hf, Wf] or [B, 1, Hf, Wf]
    step:       int,
    out_dir:    str = "debug_viz/warp",
    batch_idx:  int = 0,        # 每個 batch 只存第幾個樣本
    max_saves:  int = 200,      # 超過後停止儲存，避免磁碟爆滿
):
    """
    主要呼叫介面。

    6 格並排：
      [惡劣天氣] | [晴天原圖] | [晴天 warp 到惡劣視角] | [差異圖] | [confidence] | [flow 向量場]
    """
    # 防止無限儲存
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    if existing >= max_saves:
        return

    # 取第 batch_idx 個樣本
    def _pick(t, idx):
        return t[idx] if t.dim() == 4 or (t.dim() == 3 and t.shape[0] == 2) else t

    if img_curr.dim() == 4:
        img_curr_s   = img_curr[batch_idx].cpu()
        img_ref_s    = img_ref[batch_idx].cpu()
        flow_s       = flow[batch_idx].cpu()
        confidence_s = confidence[batch_idx].cpu()
    else:
        img_curr_s, img_ref_s = img_curr.cpu(), img_ref.cpu()
        flow_s, confidence_s  = flow.cpu(), confidence.cpu()

    # Warp 晴天影像到惡劣天氣視角
    warped_img = _warp_image(img_ref_s, flow_s)

    # 轉 uint8
    arr_curr   = _to_uint8(img_curr_s)
    arr_ref    = _to_uint8(img_ref_s)
    arr_warped = _to_uint8(warped_img)

    # 差異圖（|curr - warped| 在亮度空間）
    diff = np.abs(arr_curr.astype(np.float32) - arr_warped.astype(np.float32))
    diff_vis = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

    # Confidence map → 熱力圖
    conf_np = confidence_s.squeeze().float().numpy()   # [Hf, Wf]
    conf_np = np.clip(conf_np, 0, 1)

    # Flow 向量場（顏色輪）
    flow_np = flow_s.float().numpy()                   # [2, Hf, Wf]
    mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    ang = np.arctan2(flow_np[1], flow_np[0])           # [-π, π]
    # HSV：色相 = 方向，飽和度 = 1，明度 = 歸一化 magnitude
    hsv = np.zeros((*mag.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)          # hue 0-1
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag / (mag.max() + 1e-6)
    flow_rgb = mcolors.hsv_to_rgb(hsv)                 # [Hf, Wf, 3]

    # 繪圖
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    titles = [
        f'Adverse Weather\n(input)',
        f'Clear Weather\n(reference)',
        f'Clear → Warped\n(aligned to adverse)',
        f'|Adverse − Warped|\n(diff)',
        f'UAWarpC Confidence\nmean={conf_np.mean():.3f}',
        f'Flow Field\n(color=direction, bright=magnitude)',
    ]
    images = [arr_curr, arr_ref, arr_warped, diff_vis, conf_np, flow_rgb]
    cmaps  = [None,     None,    None,       'hot',    'RdYlGn', None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        if cmap is not None:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap == 'RdYlGn' else None)
        else:
            ax.imshow(img)

    # flow 大小統計
    fig.suptitle(
        f'Step {step} | flow_mag mean={mag.mean():.2f} max={mag.max():.2f} feat_px  '
        f'(1 feat_px = 16 real_px)',
        fontsize=9, y=1.01
    )

    save_path = os.path.join(out_dir, f'step_{step:06d}_b{batch_idx}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
