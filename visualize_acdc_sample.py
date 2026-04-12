"""
ACDC CSV 隨機抽樣視覺化
隨機取 N 筆，每筆並排顯示：
  Col 1: image_path      (adverse weather RGB)
  Col 2: ref_mask_path   (clear-weather color label → MaskEncoder 輸入)
  Col 3: gt_path         (labelTrainIds grayscale → 上色後顯示)
輸出存成 PNG，方便直接查看。
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

CSV_PATH  = "/home/rvl1421/SAM_research-1/Datasets/ref_complete.csv"
N_SAMPLES = 5       # 抽幾筆
SEED      = 42      # 固定 seed，重現用；設 None 則每次不同
OUT_PATH  = "/home/rvl1421/SAM_research-1/acdc_sample_vis.png"

# Cityscapes 19-class 調色盤 (trainId → RGB)
PALETTE = {
    0:  (128, 64,128),   # road
    1:  (244, 35,232),   # sidewalk
    2:  ( 70, 70, 70),   # building
    3:  (102,102,156),   # wall
    4:  (190,153,153),   # fence
    5:  (153,153,153),   # pole
    6:  (250,170, 30),   # traffic light
    7:  (220,220,  0),   # traffic sign
    8:  (107,142, 35),   # vegetation
    9:  (152,251,152),   # terrain
    10: ( 70,130,180),   # sky
    11: (220, 20, 60),   # person
    12: (255,  0,  0),   # rider
    13: (  0,  0,142),   # car
    14: (  0,  0, 70),   # truck
    15: (  0, 60,100),   # bus
    16: (  0, 80,100),   # train
    17: (  0,  0,230),   # motorcycle
    18: (119, 11, 32),   # bicycle
}
ID_TO_NAME = {
    0:"road",1:"sidewalk",2:"building",3:"wall",4:"fence",
    5:"pole",6:"traffic light",7:"traffic sign",8:"vegetation",
    9:"terrain",10:"sky",11:"person",12:"rider",13:"car",
    14:"truck",15:"bus",16:"train",17:"motorcycle",18:"bicycle"
}

def colorize_gt(gt_arr):
    """將 labelTrainIds 灰階圖上色成 RGB，255 顯示為黑色。"""
    h, w = gt_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in PALETTE.items():
        rgb[gt_arr == cls_id] = color
    return rgb

# ── 讀取 CSV ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
if SEED is not None:
    random.seed(SEED)
indices = random.sample(range(len(df)), min(N_SAMPLES, len(df)))
samples = df.iloc[indices].reset_index(drop=True)

print(f"抽樣 {len(samples)} 筆（seed={SEED}）")

# ── 繪圖 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(samples), 3, figsize=(18, 5 * len(samples)))
if len(samples) == 1:
    axes = axes[np.newaxis, :]

col_titles = ["image_path\n(adverse RGB)", "ref_mask_path\n(clear-weather color label)", "gt_path\n(labelTrainIds colorized)"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold')

for row_idx, (_, sample) in enumerate(samples.iterrows()):
    # --- Col 1: adverse image ---
    img = np.array(Image.open(sample['image_path']).convert('RGB'))
    axes[row_idx, 0].imshow(img)
    axes[row_idx, 0].axis('off')
    axes[row_idx, 0].set_ylabel(f"sample {indices[row_idx]}", fontsize=9)

    # --- Col 2: ref mask (color) ---
    ref = np.array(Image.open(sample['ref_mask_path']).convert('RGB'))
    axes[row_idx, 1].imshow(ref)
    axes[row_idx, 1].axis('off')

    # --- Col 3: gt (labelTrainIds → colorized) ---
    gt_arr = np.array(Image.open(sample['gt_path']))
    gt_rgb = colorize_gt(gt_arr)
    unique_ids = [v for v in np.unique(gt_arr) if v != 255]
    axes[row_idx, 2].imshow(gt_rgb)
    axes[row_idx, 2].axis('off')

    # 在 gt 圖下方標示出現的類別
    legend_patches = [
        mpatches.Patch(color=[c/255 for c in PALETTE[i]], label=f"{i}:{ID_TO_NAME[i]}")
        for i in unique_ids if i in PALETTE
    ]
    if legend_patches:
        axes[row_idx, 2].legend(
            handles=legend_patches, loc='lower left',
            fontsize=6, ncol=3, framealpha=0.7
        )

    # 在每列左側印出路徑摘要（檔名部分）
    for col, key in enumerate(['image_path', 'ref_mask_path', 'gt_path']):
        fname = sample[key].split('/')[-1]
        axes[row_idx, col].set_xlabel(fname, fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=100, bbox_inches='tight')
print(f"視覺化結果已儲存至: {OUT_PATH}")
plt.close()
