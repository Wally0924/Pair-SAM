"""
v15 ACDC val inference + visualization (Paper Protocol)
=======================================================
與 scripts/eval/eval_e1_acdc_val_paper.py 完全相同的口徑：
  * 模型 forward 在 1024x1024 解析度
  * pred / GT / invalid_mask 全部在 ACDC 原始 1080x1920 比對
  * GT、invalid_mask、input image、clear reference 都直接從 CSV 路徑讀原始 PNG
  * per-image mIoU 與 E1-paper（CMA / Refign ablation 對齊）同口徑

輸出：
  inference_viz_acdc_v15_E27_paper/result_XXX.png    每張 2x2 視覺化
  終端：per-image mIoU + 最終 SegMetricsCalculator 報表
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent / 'scripts' / 'eval'))
from _eval_common import (  # noqa: E402
    load_v15_model, build_acdc_val_loader, make_batched_input,
    CITYSCAPES_CLASSES, CITYSCAPES_PALETTE, colorize_19class,
)
from utils.seg_metrics import SegMetricsCalculator  # noqa: E402

NUM_CLASSES = 19
IGNORE_INDEX = 255


class InferenceRunner:
    def __init__(self, model, device, csv_df, output_dir="inference_results"):
        self.model = model
        self.device = device
        self.csv_df = csv_df  # 與 loader 順序一致 (shuffle=False)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = SegMetricsCalculator(classes=CITYSCAPES_CLASSES)

    @torch.no_grad()
    def predict_native(self, batch, target_hw):
        """forward → fused logits → 上採至 target_hw → argmax。"""
        batched_input = make_batched_input(batch, self.device)
        outputs = self.model(batched_input)
        low_res = outputs[0]['low_res_logits'].squeeze(0)   # (K, 256, 256)
        class_ids = outputs[0]['class_ids']

        full = torch.full(
            (1, NUM_CLASSES, 256, 256), -10.0,
            device=self.device, dtype=low_res.dtype,
        )
        for k, c in enumerate(class_ids):
            full[0, c] = low_res[k]

        fused = self.model.context_fusion_head(full)        # (1, 19, 256, 256)
        fused_hr = F.interpolate(
            fused, size=target_hw, mode='bilinear', align_corners=False,
        )
        return fused_hr.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    @staticmethod
    def load_native_assets(row):
        """從 CSV row 讀原始解析度資產：input RGB / clear RGB / GT / invalid。"""
        img = cv2.imread(str(row['image_path']), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        ref = cv2.imread(str(row['ref_image_path']), cv2.IMREAD_COLOR)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB) if ref is not None else None
        gt = cv2.imread(str(row['gt_path']), cv2.IMREAD_GRAYSCALE).astype(np.int64)

        inv_path = row.get('invalid_mask') if 'invalid_mask' in row else None
        if inv_path and Path(str(inv_path)).is_file():
            inv = cv2.imread(str(inv_path), cv2.IMREAD_GRAYSCALE) != 0
        else:
            inv = np.zeros_like(gt, dtype=bool)
        return img, ref, gt, inv

    def visualize(self, img, ref, pred, gt, idx, miou, condition):
        """2x2 grid 在原始解析度顯示。"""
        H, W = pred.shape

        # 對齊 ref / img 解析度（理論上都是 1080x1920，但保險起見 resize）
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if ref is not None and ref.shape[:2] != (H, W):
            ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)

        pred_color = colorize_19class(pred)
        gt_color = colorize_19class(gt) if gt is not None else np.zeros_like(pred_color)

        # 1080x1920 直接畫，figsize 對齊 16:9
        fig = plt.figure(figsize=(20, 12))
        title = f"Sample {idx:03d} ({condition})"
        if miou is not None:
            title += f" | mIoU: {miou:.4f}"
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.15], figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img); ax1.set_title("Input Image (Adverse)", fontsize=14); ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        if ref is not None:
            ax2.imshow(ref)
        ax2.set_title("Clear-Weather Reference", fontsize=14); ax2.axis('off')

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(pred_color); ax3.set_title("Prediction (WeatherSAM v15)", fontsize=14); ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(gt_color); ax4.set_title("Ground Truth", fontsize=14); ax4.axis('off')

        ax_legend = fig.add_subplot(gs[2, :])
        ax_legend.axis('off')
        unique = set(np.unique(pred).tolist())
        if gt is not None:
            unique.update(np.unique(gt).tolist())
        patches = []
        for cls_id in sorted(unique):
            if cls_id >= NUM_CLASSES:
                continue
            patches.append(mpatches.Patch(
                color=CITYSCAPES_PALETTE[cls_id] / 255.0,
                label=CITYSCAPES_CLASSES[cls_id],
            ))
        if patches:
            ax_legend.legend(
                handles=patches, loc='center',
                ncol=min(len(patches), 8),
                frameon=False, fontsize='medium', title="Classes Present",
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(self.output_dir / f"result_{idx:03d}.png", dpi=90)
        plt.close()

    def run_inference(self, loader, num_samples=None):
        n = 0
        pbar = tqdm(loader, desc="Inference (paper protocol)")
        for batch in pbar:
            row = self.csv_df.iloc[n]
            img, ref, gt, inv = self.load_native_assets(row)
            H, W = gt.shape  # ACDC: 1080 x 1920

            pred = self.predict_native(batch, target_hw=(H, W))

            # GT with invalid → 255
            gt_used = gt.copy()
            gt_used[inv] = IGNORE_INDEX

            condition = str(row['condition']) if 'condition' in row else 'unknown'

            miou = SegMetricsCalculator.compute_image_miou(pred, gt_used, NUM_CLASSES)
            self.metrics.update(pred, gt_used, condition=condition)
            print(f"📊 Image {n:03d} ({condition:5s}) | mIoU: {miou:.4f}")

            self.visualize(img, ref, pred, gt_used, idx=n, miou=miou, condition=condition)

            n += 1
            if num_samples is not None and n >= num_samples:
                break

        if n > 0:
            self.metrics.print_report(self.metrics.compute())


if __name__ == "__main__":
    CHECKPOINT_PATH = str(
        _THIS.parent /
        "outputs_weather_sam_mask2former_testv15" /
        "weather_sam_best_latest.pth"
    )
    TEST_CSV_PATH = str(
        _THIS.parent.parent / "Datasets" / "acdc_adverse_ref_rgb_val.csv"
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "inference_viz_acdc_v15_paper"

    print(f"Loading v15 model: {Path(CHECKPOINT_PATH).name}")
    model = load_v15_model(CHECKPOINT_PATH, device=DEVICE)

    print(f"Building ACDC val loader: {Path(TEST_CSV_PATH).name}")
    loader = build_acdc_val_loader(TEST_CSV_PATH, batch_size=1, num_workers=4)
    csv_df = loader.dataset.data.reset_index(drop=True)

    print(f"Output dir: {OUTPUT_DIR}  (paper protocol: native 1080x1920)\n")
    runner = InferenceRunner(model, DEVICE, csv_df, output_dir=OUTPUT_DIR)
    runner.run_inference(loader, num_samples=None)
