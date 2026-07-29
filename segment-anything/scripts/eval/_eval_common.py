# segment-anything/scripts/eval/_eval_common.py
"""共用工具：ablation config 驅動的模型載入、ACDC val dataloader、Cityscapes 19-class 調色盤。"""
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

from utils.pair_dataloader import PairSegmentationDataset


# ── 預設路徑 ───────────────────────────────────────────────
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


def load_pair_sam_from_ablation(ckpt_path, config_path=None, device='cuda'):
    """依 ablation_config.json 重建模型並載入 ckpt，確保與訓練 config 完全一致。

    複用 build_pair_sam_from_config 的 checkpoint 載入（自動處理 model_state_dict
    key 與 shape 過濾），並由 cfg 設定 decoder_mode / use_lrh / use_reference / inject。
    回傳 (model, cfg)。
    """
    import json
    from segment_anything.build_pair_sam import build_pair_sam_from_config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(ckpt_path), 'ablation_config.json')
    with open(config_path) as f:
        cfg = json.load(f)
    model = build_pair_sam_from_config(cfg, checkpoint=ckpt_path)
    return model.to(device).eval(), cfg


def build_acdc_val_loader(csv_path: str = DEFAULT_VAL_CSV,
                           batch_size: int = 1, num_workers: int = 2):
    """ACDC val DataLoader，batch_size=1 簡化 per-condition 統計。"""
    ds = PairSegmentationDataset(
        csv_file=csv_path, image_size=1024, mode='val', force_raw_images=True,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=PairSegmentationDataset.collate_fn,
    )


def make_batched_input(batch: dict, device: str) -> list:
    """把 dataloader 出來的 batch dict 轉成 PairSAM.forward 需要的 list[dict]。
    與 pair_trainer.py 中的轉換邏輯保持一致。
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
