# segment-anything/scripts/eval/viz_e5_warp_confidence.py
"""
E5: UAWarpC warp 與 confidence 可視化
4 條件各 1 張樣本 × 4 欄：
    clear reference / warped reference / warped × confidence / adverse
對應論文：Refign Fig.7
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (
    load_v15_model, pick_first_per_condition, denorm_image,
    CONDITION_NAMES, OUTPUT_ROOT, DEFAULT_CKPT, DEFAULT_VAL_CSV,
)

DEVICE = 'cuda'


def upsample_flow(flow_lr: torch.Tensor, out_hw: tuple) -> torch.Tensor:
    """把低解析度 flow (B, 2, H_lr, W_lr) 上採至 (B, 2, H_hi, W_hi)。
    flow 的數值是 pixel-offset 單位，所以同步乘上空間縮放係數。
    """
    B, _, H_lr, W_lr = flow_lr.shape
    H_hi, W_hi = out_hw
    flow_hi = F.interpolate(
        flow_lr, size=out_hw, mode='bilinear', align_corners=False,
    )
    sx = float(W_hi) / float(W_lr)
    sy = float(H_hi) / float(H_lr)
    flow_hi[:, 0] = flow_hi[:, 0] * sx
    flow_hi[:, 1] = flow_hi[:, 1] * sy
    return flow_hi


def main():
    from segment_anything.modeling.cma_utils import warp
    from utils.weather_dataloader import WeatherSegmentationDataset

    model = load_v15_model(DEFAULT_CKPT, device=DEVICE)
    ds = WeatherSegmentationDataset(
        csv_file=DEFAULT_VAL_CSV, image_size=1024, mode='val', force_raw_images=True,
    )
    picked = pick_first_per_condition(DEFAULT_VAL_CSV)
    print('Picked indices:', picked)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    column_titles = ['Clear Reference', 'Warped Reference',
                     'Warped × Confidence', 'Adverse (Target)']

    with torch.no_grad():
        for row, cid in enumerate(CONDITION_NAMES):
            idx = picked[cid]
            item = ds[idx]
            adverse = item['image'].unsqueeze(0).to(DEVICE)         # (1, 3, 1024, 1024)
            clear   = item['clear_image'].unsqueeze(0).to(DEVICE)

            # 1. 呼叫 pre_align 取得 flow + confidence（內部 out_size=(64, 64)）
            _ = model.fusion_module.pre_align(adverse, clear)
            flow_lr = model.fusion_module._last_flow.to(DEVICE)               # (1, 2, 64, 64)
            conf_lr = model.fusion_module._last_confidence_map.to(DEVICE)     # (1, 1, 64, 64)

            # 2. 上採 flow 至 1024×1024（同時 scaling pixel-offset 值）
            flow_hi = upsample_flow(flow_lr, out_hw=(1024, 1024))

            # 3. 用 flow 把 clear reference 影像 warp 至 adverse 視角
            warped_clear, valid = warp(clear, flow_hi, return_mask=True)      # (1, 3, 1024, 1024)

            # 4. 上採 confidence 至 1024×1024
            conf_hi = F.interpolate(conf_lr, size=(1024, 1024), mode='bilinear',
                                     align_corners=False)
            conf_hi = conf_hi.clamp(0.0, 1.0)                                   # (1, 1, 1024, 1024)
            conf_2d = conf_hi.squeeze(0).squeeze(0).cpu().numpy()              # (1024, 1024)

            # 5. warped_with_conf：低 confidence 區域 fade 為白色
            warped_np = denorm_image(warped_clear[0]).astype(np.float32)       # (H, W, 3) 0~255
            white = np.ones_like(warped_np) * 255.0
            alpha = conf_2d[..., None]                                          # (H, W, 1)
            warped_with_conf = (warped_np * alpha + white * (1.0 - alpha)).astype(np.uint8)

            # 渲染
            axes[row, 0].imshow(denorm_image(clear[0]))
            axes[row, 1].imshow(denorm_image(warped_clear[0]))
            axes[row, 2].imshow(warped_with_conf)
            axes[row, 3].imshow(denorm_image(adverse[0]))

            axes[row, 0].set_ylabel(
                CONDITION_NAMES[cid].capitalize(),
                fontsize=14, fontweight='bold',
            )
            for col in range(4):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / 'e5_warp_confidence.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Figure written: {out_path}')


if __name__ == '__main__':
    main()
