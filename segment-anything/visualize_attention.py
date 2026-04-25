"""
visualize_attention.py

DeformableCrossViewAlignment 視覺化工具。

每個 sample 輸出一張多面板圖，包含：
  Row 0 : 退化影像 | 晴天參考 | 採樣點密度圖（哪些 f_ref 位置被最多 query 採樣）
  Row 1 : Per-head offset magnitude 長條圖 | Per-head K-point entropy 長條圖
  Row 2 : 各語意類別代表 patch 的採樣點分布（K=4 個點疊加在 f_ref 上，不同 head 不同顏色）

用法：
    cd /home/rvl1421/SAM_research-1/segment-anything
    python visualize_attention.py [--n_samples N] [--output_dir DIR] [--checkpoint PATH]
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import cv2
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from segment_anything.build_weather_sam import build_weather_sam_vit_h
from utils.weather_dataloader import WeatherSegmentationDataset

# ── 常數 ────────────────────────────────────────────────────────

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

CLASS_MAP = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
    "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
    "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
    "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18,
}

PROBE_CLASSES = ["road", "sky", "building", "vegetation", "car"]
FEAT_SIZE = 64   # ViT-H neck 空間解析度（H = W = 64）

# 每個 head 使用不同顏色，方便區分 8 個 head 的採樣點
HEAD_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


# ── DeformableCapture ────────────────────────────────────────────

class DeformableCapture:
    """
    透過 monkey-patch 攔截 DeformableCrossViewAlignment.forward，
    擷取採樣點座標與注意力權重供視覺化使用。

    forward 結束後可讀取：
        sampling_locs : np.ndarray (N, num_heads, K, 2)
            每個 query × head 的 K 個採樣點，grid_sample 座標 [-1, 1]，
            格式為 (x=col, y=row)。
        offsets       : np.ndarray (N, num_heads, K, 2)
            從 identity 位置出發的偏移量，範圍 [-0.5, 0.5]。
        attn_weights  : np.ndarray (N, num_heads, K)
            softmax 後的 K-point 注意力權重，每組加總為 1。
    """

    def __init__(self):
        self.sampling_locs: np.ndarray = None   # (N, H, K, 2)
        self.offsets: np.ndarray = None          # (N, H, K, 2)
        self.attn_weights: np.ndarray = None     # (N, H, K)
        self._original_forward = None

    def register(self, module):
        """Monkey-patch module.forward，保留原始 forward 以便還原。"""
        self._original_forward = module.forward
        capture = self

        def patched_forward(f_curr, f_ref):
            B, C, H, W = f_curr.shape
            N = H * W

            # ── 與原始 forward 完全相同的計算 ──────────────────
            q_flat = f_curr.flatten(2).transpose(1, 2)     # (B, N, C)
            q = module.q_proj(q_flat)

            offsets_raw = module.offset_net(q)
            offsets = offsets_raw.view(B, N, module.num_heads, module.num_points, 2)
            offsets = offsets.tanh() * 0.5

            ref_grid = module._build_ref_grid(H, W, f_curr.device)
            sampling_locs = (ref_grid + offsets).clamp(-1.0, 1.0)

            v_flat = module.v_proj(f_ref.flatten(2).transpose(1, 2))
            v_spatial = v_flat.transpose(1, 2).view(B, C, H, W)

            attn_weights = module.attn_weight_net(q)
            attn_weights = attn_weights.view(B, N, module.num_heads, module.num_points)
            attn_weights = attn_weights.softmax(dim=-1)

            # ── 擷取診斷張量（batch=0，detach 到 CPU）──────────
            with torch.no_grad():
                # sampling_locs: (B, N, H, K, 2) → 取 batch 0 → (N, H, K, 2)
                capture.sampling_locs = sampling_locs[0].detach().cpu().float().numpy()
                capture.offsets = offsets[0].detach().cpu().float().numpy()
                capture.attn_weights = attn_weights[0].detach().cpu().float().numpy()

            # ── 繼續原始 forward 後半段 ─────────────────────────
            head_outputs = []
            for h_idx in range(module.num_heads):
                v_h = v_spatial[:, h_idx * module.head_dim:(h_idx + 1) * module.head_dim]
                locs_h = sampling_locs[:, :, h_idx, :, :]
                locs_h = locs_h.reshape(B, N * module.num_points, 1, 2)
                sampled = F.grid_sample(
                    v_h, locs_h,
                    mode="bilinear", padding_mode="border", align_corners=True,
                )
                sampled = sampled.squeeze(-1).permute(0, 2, 1)
                sampled = sampled.view(B, N, module.num_points, module.head_dim)
                w_h = attn_weights[:, :, h_idx, :].unsqueeze(-1)
                head_outputs.append((w_h * sampled).sum(dim=2))

            out = torch.cat(head_outputs, dim=-1)
            out = module.out_proj(out)
            f_align = out.transpose(1, 2).view(B, C, H, W)
            f_align = module.norm(f_align)

            # 同步更新模組內建的監控屬性
            with torch.no_grad():
                module._last_offset_mag = float(offsets.detach().abs().mean().item())
                import math
                a = attn_weights.float().mean(dim=[0, 1])
                ent = -(a * (a + 1e-12).log()).sum(dim=-1)
                max_ent = math.log(module.num_points)
                module._last_attn_entropy = float((ent / max_ent).mean().item())

            return f_align

        module.forward = patched_forward

    def restore(self, module):
        """還原原始 forward。"""
        if self._original_forward is not None:
            module.forward = self._original_forward
            self._original_forward = None


# ── 輔助函式 ────────────────────────────────────────────────────

def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float/uint8 tensor → (H, W, 3) float32 [0, 1]."""
    arr = t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def normalized_to_pixel(coord_xy: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    grid_sample 正規化座標 [-1, 1] → 影像像素座標。
    coord_xy: (..., 2)，格式為 (x=col, y=row)。
    回傳 (..., 2)，格式為 (px_col, px_row)。
    """
    px_col = (coord_xy[..., 0] + 1.0) / 2.0 * (img_w - 1)
    px_row = (coord_xy[..., 1] + 1.0) / 2.0 * (img_h - 1)
    return np.stack([px_col, px_row], axis=-1)


def build_sampling_density_map(
    sampling_locs: np.ndarray,    # (N, num_heads, K, 2) in [-1, 1]
    feat_size: int = FEAT_SIZE,
) -> np.ndarray:
    """
    統計所有 query × head × K 點落在 f_ref 的哪些格子，
    回傳 (feat_size, feat_size) density map（正規化到 [0, 1]）。
    """
    density = np.zeros((feat_size, feat_size), dtype=np.float32)
    N, num_heads, K, _ = sampling_locs.shape

    # 轉換到 feature map 的 grid 座標
    col_idx = np.clip(
        ((sampling_locs[..., 0] + 1.0) / 2.0 * (feat_size - 1)).astype(int),
        0, feat_size - 1
    )
    row_idx = np.clip(
        ((sampling_locs[..., 1] + 1.0) / 2.0 * (feat_size - 1)).astype(int),
        0, feat_size - 1
    )
    np.add.at(density, (row_idx.ravel(), col_idx.ravel()), 1)

    lo, hi = density.min(), density.max()
    return (density - lo) / (hi - lo + 1e-8)


def overlay_heatmap(
    img_rgb: np.ndarray,
    hmap: np.ndarray,
    alpha: float = 0.55,
    colormap=plt.cm.jet,
) -> np.ndarray:
    """
    img_rgb : (H, W, 3) float32
    hmap    : (feat_size, feat_size) float32 [0, 1]
    """
    h, w = img_rgb.shape[:2]
    hmap_up = cv2.resize(hmap, (w, h), interpolation=cv2.INTER_LINEAR)
    color = colormap(hmap_up)[..., :3]
    return np.clip((1 - alpha) * img_rgb + alpha * color, 0.0, 1.0)


def select_probe_patches(gt_np: np.ndarray) -> dict:
    """
    從 GT mask 在 feat_size × feat_size 的 feature map 上
    為每個目標類別選出代表 patch（centroid 最近的格子）。

    Returns: {class_name: (flat_idx, feat_row, feat_col)}
    """
    gt_small = cv2.resize(
        gt_np.astype(np.uint8), (FEAT_SIZE, FEAT_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    result = {}
    for cls_name in PROBE_CLASSES:
        cls_id = CLASS_MAP[cls_name]
        mask = (gt_small == cls_id)
        if mask.sum() == 0:
            continue
        rows, cols = np.where(mask)
        cr = int(np.clip(rows.mean(), 0, FEAT_SIZE - 1))
        cc = int(np.clip(cols.mean(), 0, FEAT_SIZE - 1))
        flat_idx = cr * FEAT_SIZE + cc
        result[cls_name] = (flat_idx, cr, cc)
    return result


# ── 視覺化核心 ───────────────────────────────────────────────────

def visualize_sample(
    idx: int,
    deg_img: np.ndarray,
    ref_img: np.ndarray,
    capture: DeformableCapture,
    probe_patches: dict,
    condition: str,
    output_dir: str,
):
    """
    生成並儲存單張 sample 的 DeformableCrossViewAlignment 視覺化圖。

    版面（由上至下）：
      Row 0 : 退化影像 | 晴天參考 | 全局採樣點密度圖（疊加在 f_ref 上）
      Row 1 : Per-head offset magnitude 長條圖 | Per-head K-point entropy 長條圖
      Row 2 : 各語意類別的 probe patch 採樣點分布（K×H 個點疊在 f_ref 上）
    """
    sampling_locs = capture.sampling_locs  # (N, num_heads, K, 2)
    offsets       = capture.offsets        # (N, num_heads, K, 2)
    attn_weights  = capture.attn_weights   # (N, num_heads, K)

    num_heads = sampling_locs.shape[1]
    K         = sampling_locs.shape[2]
    img_h, img_w = ref_img.shape[:2]

    # ── 採樣點密度圖 ────────────────────────────────────────────
    density_map = build_sampling_density_map(sampling_locs)
    density_overlay = overlay_heatmap(ref_img, density_map, alpha=0.6, colormap=plt.cm.hot)

    # ── Per-head 統計 ────────────────────────────────────────────
    # offset magnitude per head: (num_heads,)
    head_offset_mag = np.abs(offsets).mean(axis=(0, 2, 3))   # mean over N, K, xy

    # entropy over K points per head: (num_heads,)
    # attn_weights: (N, num_heads, K)
    mean_w = attn_weights.mean(axis=0)   # (num_heads, K)
    head_entropy = []
    max_ent = np.log(K)
    for h_idx in range(num_heads):
        w = mean_w[h_idx]
        w = w / (w.sum() + 1e-8)
        ent = float(-np.sum(w * np.log(w + 1e-12)))
        head_entropy.append(ent / max_ent)
    head_entropy = np.array(head_entropy)

    n_probes = max(len(probe_patches), 1)

    # ── GridSpec ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(n_probes * 4, 14), 14), dpi=110)
    fig.suptitle(
        f"Sample {idx:03d}  |  condition: {condition}  |  DeformableCrossViewAlignment",
        fontsize=13, y=0.99,
    )

    gs = gridspec.GridSpec(
        3, max(n_probes, 3),
        figure=fig,
        height_ratios=[1.6, 1.0, 1.6],
        hspace=0.38, wspace=0.25,
    )

    # ── Row 0: 退化 | 晴天 | 採樣點密度 ─────────────────────────
    ax_deg = fig.add_subplot(gs[0, 0])
    ax_deg.imshow(deg_img)
    ax_deg.set_title("Degraded Image (f_curr)", fontsize=10)
    ax_deg.axis("off")

    ax_ref = fig.add_subplot(gs[0, 1])
    ax_ref.imshow(ref_img)
    ax_ref.set_title("Clear Reference (f_ref)", fontsize=10)
    ax_ref.axis("off")

    ax_density = fig.add_subplot(gs[0, 2])
    ax_density.imshow(density_overlay)
    ax_density.set_title(
        "Sampling Point Density on f_ref\n"
        "(bright = queried by many positions)",
        fontsize=10,
    )
    ax_density.axis("off")

    # ── Row 1: Per-head offset magnitude | Per-head K-entropy ────
    n_bar_cols = max(n_probes, 3)
    half = n_bar_cols // 2

    ax_off = fig.add_subplot(gs[1, :half])
    bar_colors_off = [HEAD_COLORS[h % len(HEAD_COLORS)] for h in range(num_heads)]
    ax_off.bar(range(num_heads), head_offset_mag, color=bar_colors_off,
               edgecolor="white", linewidth=0.5)
    ax_off.axhline(0.1, color="gray", linestyle="--", linewidth=1.0,
                   label="0.10 reference line")
    ax_off.set_ylim(0, 0.55)
    ax_off.set_xticks(range(num_heads))
    ax_off.set_xticklabels([f"H{i}" for i in range(num_heads)], fontsize=9)
    ax_off.set_xlabel("Head index", fontsize=9)
    ax_off.set_ylabel("Mean |offset|  [0, 0.5]", fontsize=9)
    ax_off.set_title(
        "Per-Head Offset Magnitude\n"
        "(higher = sampling points moved farther from identity)",
        fontsize=10,
    )
    ax_off.legend(fontsize=8)

    ax_ent = fig.add_subplot(gs[1, half:])
    bar_colors_ent = ["#e74c3c" if e > 0.90 else "#3498db" for e in head_entropy]
    ax_ent.bar(range(num_heads), head_entropy, color=bar_colors_ent,
               edgecolor="white", linewidth=0.5)
    ax_ent.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.2,
                   label="Uniform over K (entropy=1.0)")
    ax_ent.axhline(0.5, color="#f39c12", linestyle=":", linewidth=1.0,
                   label="50% threshold")
    ax_ent.set_ylim(0, 1.15)
    ax_ent.set_xticks(range(num_heads))
    ax_ent.set_xticklabels([f"H{i}" for i in range(num_heads)], fontsize=9)
    ax_ent.set_xlabel("Head index", fontsize=9)
    ax_ent.set_ylabel("Entropy / log(K)", fontsize=9)
    ax_ent.set_title(
        "Per-Head K-point Attention Entropy\n"
        "(red = K weights nearly uniform; blue = concentrated on 1~2 points)",
        fontsize=10,
    )
    ax_ent.legend(fontsize=8)

    # ── Row 2: Per-class probe patch 採樣點分布 ──────────────────
    if probe_patches:
        probe_items = list(probe_patches.items())
        for c_idx, (cls_name, (flat_idx, feat_r, feat_c)) in enumerate(probe_items):
            if c_idx >= gs.get_geometry()[1]:
                break
            ax_p = fig.add_subplot(gs[2, c_idx])
            ax_p.imshow(ref_img)

            # query 在 feature map 中的位置 → 對應影像像素
            q_ref_col = feat_c / FEAT_SIZE * img_w
            q_ref_row = feat_r / FEAT_SIZE * img_h

            # 畫出 query 在 f_curr 對應位置（白色正方形）
            cell_w = img_w / FEAT_SIZE
            cell_h = img_h / FEAT_SIZE
            rect = mpatches.FancyBboxPatch(
                (q_ref_col, q_ref_row), cell_w, cell_h,
                boxstyle="square,pad=0",
                linewidth=2, edgecolor="white", facecolor="none",
            )
            ax_p.add_patch(rect)

            # 畫出每個 head 的 K 個採樣點（各 head 顏色不同）
            # sampling_locs[flat_idx]: (num_heads, K, 2) in [-1, 1]
            locs_query = sampling_locs[flat_idx]   # (num_heads, K, 2)
            px = normalized_to_pixel(locs_query, img_h, img_w)  # (num_heads, K, 2)

            for h_idx in range(num_heads):
                col = HEAD_COLORS[h_idx % len(HEAD_COLORS)]
                for k in range(K):
                    ax_p.plot(
                        px[h_idx, k, 0], px[h_idx, k, 1],
                        marker="x", markersize=8, markeredgewidth=2,
                        color=col, alpha=0.85,
                    )

            # 圖例：每個顏色 = 一個 head
            legend_handles = [
                mpatches.Patch(color=HEAD_COLORS[h % len(HEAD_COLORS)], label=f"H{h}")
                for h in range(num_heads)
            ]
            ax_p.legend(
                handles=legend_handles,
                fontsize=6, loc="lower right",
                ncol=4, framealpha=0.6,
            )

            cls_color = CITYSCAPES_PALETTE[CLASS_MAP[cls_name]] / 255.0
            ax_p.set_title(
                f"{cls_name}  (patch {feat_r},{feat_c})\n"
                f"× = K sampling points per head in f_ref",
                fontsize=9, color=cls_color,
                bbox=dict(facecolor="black", alpha=0.55, pad=2),
            )
            ax_p.axis("off")
    else:
        ax_na = fig.add_subplot(gs[2, :])
        ax_na.text(
            0.5, 0.5,
            "No probe patches (GT unavailable or no matching classes)",
            ha="center", va="center", transform=ax_na.transAxes, fontsize=11,
        )
        ax_na.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"attn_{idx:03d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")
    return save_path


# ── 主程式 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeformableCrossViewAlignment Sampling Visualizer"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=(
            "/home/rvl1421/SAM_research-1/segment-anything/"
            "outputs_weather_sam_mask2former_testv5_noabl/"
            "weather_sam_best_latest.pth"
        ),
    )
    parser.add_argument(
        "--csv", type=str,
        default="/home/rvl1421/SAM_research-1/Datasets/acdc_val_with_embeddings.csv",
    )
    parser.add_argument("--n_samples", type=int, default=20,
                        help="要視覺化的樣本數量")
    parser.add_argument("--output_dir", type=str, default="attn_viz_deformable",
                        help="輸出圖片目錄")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── 載入模型 ──────────────────────────────────────────────────
    print(f"Loading model from: {args.checkpoint}")
    model = build_weather_sam_vit_h(checkpoint=args.checkpoint)
    model.to(args.device)
    model.eval()

    # ── 安裝 deformable capture ───────────────────────────────────
    capture = DeformableCapture()
    capture.register(model.fusion_module)

    # ── 載入資料集 ────────────────────────────────────────────────
    dataset = WeatherSegmentationDataset(csv_file=args.csv, image_size=1024, mode="val")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=WeatherSegmentationDataset.collate_fn,
    )
    print(f"Dataset size: {len(dataset)}  |  Visualizing {args.n_samples} samples")
    print(f"Output dir  : {args.output_dir}/\n")

    saved_paths = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.n_samples:
                break

            # ── 準備影像 numpy array ──────────────────────────────
            # Cache 模式（val set）不含 image tensor，從 CSV 的 image_path 直接讀取
            if "image" in batch:
                deg_img = tensor_to_rgb(batch["image"][0])
            else:
                img_path = str(dataset.data.iloc[idx]["image_path"])
                raw = cv2.imread(img_path)
                if raw is None:
                    print(f"  [WARN] Cannot load image: {img_path}. Skipping.")
                    continue
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                raw = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                deg_img = raw.astype(np.float32) / 255.0

            ref_img = tensor_to_rgb(batch["reference_mask"][0])
            gt_np   = batch["gt_mask"][0].numpy()
            condition = (batch.get("condition") or ["unknown"])[0] or "unknown"

            # ── 決定 probe patch indices ─────────────────────────
            probe_patches = select_probe_patches(gt_np)

            # ── Forward（觸發 deformable capture）────────────────
            input_record = {
                "text_prompts":  batch["text_prompts"][0] or ["road"],
                "original_size": tuple(batch["original_size"][0]),
                "clear_embedding": batch["clear_embedding"][0].to(args.device),
                "condition_id":  batch["condition_id"][0].to(args.device),
                "location":      batch["location"][0].to(args.device),
                "reference_mask": batch["reference_mask"][0].to(args.device),
                "ref_void_mask": batch["ref_void_mask"][0].to(args.device),
            }
            if "image_embedding" in batch:
                input_record["image_embedding"] = batch["image_embedding"][0].to(args.device)
            else:
                input_record["image"] = batch["image"][0].to(args.device)

            _ = model([input_record])

            if capture.sampling_locs is None:
                print(f"  [WARN] Sample {idx}: sampling locs not captured. Skipping.")
                continue

            # ── 印出摘要 ─────────────────────────────────────────
            mean_offset = float(np.abs(capture.offsets).mean())
            mean_entropy = float(
                np.array([
                    -(w * np.log(w + 1e-12)).sum() / np.log(capture.sampling_locs.shape[2])
                    for w in capture.attn_weights.mean(axis=0)
                ]).mean()
            )
            print(
                f"  Sample {idx:03d} | cond={condition:<6s} | "
                f"mean_offset={mean_offset:.4f} | "
                f"mean_K_entropy={mean_entropy:.4f} | "
                f"probes={list(probe_patches.keys())}"
            )

            # ── 視覺化 ───────────────────────────────────────────
            path = visualize_sample(
                idx=idx,
                deg_img=deg_img,
                ref_img=ref_img,
                capture=capture,
                probe_patches=probe_patches,
                condition=condition,
                output_dir=args.output_dir,
            )
            saved_paths.append(path)

    # ── 還原模型 forward ─────────────────────────────────────────
    capture.restore(model.fusion_module)

    print(f"\nDone. {len(saved_paths)} figures saved to {args.output_dir}/")
    print("\n=== 閱讀指南 ===")
    print("Row 0 密度圖（Sampling Density）：越亮代表該 f_ref 位置被越多 query 採樣")
    print("  → 若密度集中在語意相關區域（路面 query → 密度在路面），代表模組有學到空間對應")
    print("  → 若密度均勻分布，代表採樣點雖有移動（offset_mag > 0）但方向尚無語意意義")
    print("Row 1 左：Offset Magnitude — 每個 head 的偏移量，> 0.1 代表有在移動")
    print("Row 1 右：K-point Entropy — 接近 1.0 代表 4 個權重接近均勻，0.0 = 集中在 1 點")
    print("Row 2 採樣點（× 標記）：同類別 query 的 K=4 採樣位置，不同顏色 = 不同 head")
    print("  → 若同類別的 × 集中在 f_ref 語意相符區域，代表學到有意義的跨視角對應")


if __name__ == "__main__":
    main()
