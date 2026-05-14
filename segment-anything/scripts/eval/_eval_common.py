# segment-anything/scripts/eval/_eval_common.py
"""共用工具：v15 checkpoint 載入、ACDC val dataloader、Cityscapes 19-class 調色盤。"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# 讓 script 可從 segment-anything 根目錄被執行（含 utils 與 segment_anything 兩個 import path）
_THIS = Path(__file__).resolve()
_SEGANY_ROOT = _THIS.parents[2]   # .../segment-anything
if str(_SEGANY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGANY_ROOT))

from segment_anything.build_weather_sam import build_weather_sam_vit_h
from utils.weather_dataloader import WeatherSegmentationDataset


# ── 預設路徑 ───────────────────────────────────────────────
DEFAULT_CKPT = str(
    _SEGANY_ROOT / "outputs_weather_sam_mask2former_testv15" /
    "best_E27_mIoU65.68_LR4.0e-05.pth"
)
DEFAULT_VAL_CSV = str(
    _SEGANY_ROOT.parent / "Datasets" / "acdc_adverse_ref_rgb_val.csv"
)
OUTPUT_ROOT = _SEGANY_ROOT.parent / "docs" / "experiments" / "v15-eval-2026-05-14"

# ── ACDC condition 對照 ────────────────────────────────────
CONDITION_NAMES = {0: 'fog', 1: 'rain', 2: 'snow', 3: 'night'}

# ── Cityscapes 19-class 標準調色盤（與 ACDC GT trainIds 對齊）──
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],  # road
    [244,  35, 232],  # sidewalk
    [ 70,  70,  70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170,  30],  # traffic light
    [220, 220,   0],  # traffic sign
    [107, 142,  35],  # vegetation
    [152, 251, 152],  # terrain
    [ 70, 130, 180],  # sky
    [220,  20,  60],  # person
    [255,   0,   0],  # rider
    [  0,   0, 142],  # car
    [  0,   0,  70],  # truck
    [  0,  60, 100],  # bus
    [  0,  80, 100],  # train
    [  0,   0, 230],  # motorcycle
    [119,  11,  32],  # bicycle
], dtype=np.uint8)


def colorize_19class(mask: np.ndarray) -> np.ndarray:
    """19-class trainID mask (H, W) → RGB (H, W, 3)。255 視為 ignore，輸出黑色。"""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id in range(19):
        color[mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
    return color


def load_v15_model(ckpt_path: str = DEFAULT_CKPT, device: str = 'cuda'):
    """載入 best_E18 v15 checkpoint，啟用 pre-hook cross-attn adapter。

    strict=False：v5 後續加了 `_last_kv_keep_ratio` 等 buffer，舊 ckpt 不含此欄。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_weather_sam_vit_h(num_classes=19, checkpoint=None)
    state = torch.load(ckpt_path, map_location='cpu')
    # 容忍多種儲存格式
    if isinstance(state, dict) and 'model_state_dict' in state:
        sd = state['model_state_dict']
    elif isinstance(state, dict) and 'model' in state:
        sd = state['model']
    else:
        sd = state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[load_v15_model] 忽略 {len(unexpected)} 個多餘鍵（OK）")
    if missing:
        print(f"[load_v15_model] 缺少 {len(missing)} 個鍵（多為新增 buffer，OK）")
    model.enable_vgg_adapter('pre')
    return model.to(device).eval()


def build_acdc_val_loader(csv_path: str = DEFAULT_VAL_CSV,
                           batch_size: int = 1, num_workers: int = 2):
    """ACDC val DataLoader，batch_size=1 簡化 per-condition 統計。"""
    ds = WeatherSegmentationDataset(
        csv_file=csv_path, image_size=1024, mode='val', force_raw_images=True,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=WeatherSegmentationDataset.collate_fn,
    )


def make_batched_input(batch: dict, device: str) -> list:
    """把 dataloader 出來的 batch dict 轉成 WeatherSAM.forward 需要的 list[dict]。
    與 weather_trainer.py 中的轉換邏輯保持一致。
    """
    bs = batch['gt_mask'].shape[0]
    out = []
    for i in range(bs):
        item = {
            'image':        batch['image'][i].to(device),
            'clear_image':  batch['clear_image'][i].to(device),
            'text_prompts': batch['text_prompts'][i],
            'original_size': batch['original_size'][i],
            'condition_id': batch['condition_id'][i],
        }
        out.append(item)
    return out


def pick_first_per_condition(csv_path: str = DEFAULT_VAL_CSV) -> dict:
    """從 ACDC val csv 為每個 condition_id 找出首個 row 的索引（CSV 內順序）。
    回傳：{0: idx_fog, 1: idx_rain, 2: idx_snow, 3: idx_night}
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    picked = {}
    for cid in CONDITION_NAMES:
        rows = df.index[df['condition_id'] == cid].tolist()
        if not rows:
            raise ValueError(f'No sample for condition_id={cid}')
        picked[cid] = rows[0]
    return picked


def denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    """把 dataloader 輸出（0~255 範圍 float tensor）轉成 matplotlib 友善的 uint8 RGB。"""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
