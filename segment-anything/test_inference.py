"""
v15 ACDC val inference + visualization
======================================
與 scripts/eval/eval_e1_acdc_val_full.py 相同的 forward + post-process pipeline，
但聚焦在 per-image 2x2 可視化 + per-image mIoU 列印，最後彙總 per-class /
per-condition metrics。所有評估與可視化皆在 1024x1024 解析度上進行
（與 weather_trainer.validate_epoch 一致）。
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

# 複用 scripts/eval/_eval_common.py 內的 v15 載入/dataloader/調色盤工具
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
    def __init__(self, model, device, output_dir="inference_results"):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = SegMetricsCalculator(classes=CITYSCAPES_CLASSES)

    @torch.no_grad()
    def predict_single_image(self, batch) -> np.ndarray:
        """v15 forward → scatter → context_fusion_head → 1024 argmax。

        與 scripts/eval/eval_e1_acdc_val_full.py 邏輯 1:1 對齊。
        """
        batched_input = make_batched_input(batch, self.device)
        outputs = self.model(batched_input)

        low_res = outputs[0]['low_res_logits'].squeeze(0)   # (K, 256, 256)
        class_ids = outputs[0]['class_ids']                 # List[int]

        # K=0 防呆：所有 class 填 -10，argmax 落到 class 0
        full = torch.full(
            (1, NUM_CLASSES, 256, 256), -10.0,
            device=self.device, dtype=low_res.dtype,
        )
        for k, c in enumerate(class_ids):
            full[0, c] = low_res[k]

        fused = self.model.context_fusion_head(full)        # (1, 19, 256, 256)
        fused_hr = F.interpolate(
            fused, size=(1024, 1024), mode='bilinear', align_corners=False,
        )
        pred = fused_hr.argmax(dim=1).squeeze(0)            # (1024, 1024)
        return pred.cpu().numpy().astype(np.int64)

    def prepare_gt_1024(self, batch):
        """GT mask 縮到 1024，invalid_mask 標記設 255。"""
        if 'gt_mask' not in batch:
            return None
        gt = batch['gt_mask'][0].to(self.device).long()             # (H, W)
        gt_1024 = F.interpolate(
            gt.unsqueeze(0).unsqueeze(0).float(),
            size=(1024, 1024), mode='nearest',
        ).long().squeeze(0).squeeze(0)
        inv = batch['invalid_mask'][0].to(self.device)
        if inv.any():
            inv_1024 = F.interpolate(
                inv.unsqueeze(0).unsqueeze(0).float(),
                size=(1024, 1024), mode='nearest',
            ).bool().squeeze(0).squeeze(0)
            gt_1024[inv_1024] = IGNORE_INDEX
        return gt_1024.cpu().numpy().astype(np.int64)

    def visualize(self, batch, pred, gt, idx, miou):
        """2x2 grid：Input / Clear Reference / Pred / GT，全部在 1024 顯示。"""
        # Input image: dataloader 已 letterbox 至 1024
        img = batch['image'][0].permute(1, 2, 0).cpu().numpy() / 255.0
        img = np.clip(img, 0, 1)

        ref = batch['reference_mask'][0].permute(1, 2, 0).cpu().numpy() / 255.0
        ref = np.clip(ref, 0, 1)
        if ref.shape[:2] != (1024, 1024):
            ref = cv2.resize(ref, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        pred_color = colorize_19class(pred)
        gt_color = colorize_19class(gt) if gt is not None else np.zeros_like(pred_color)

        fig = plt.figure(figsize=(16, 12))
        if miou is not None:
            fig.suptitle(
                f"Sample {idx:03d} | Image mIoU: {miou:.4f}",
                fontsize=20, fontweight='bold', y=0.98,
            )

        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.15], figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img); ax1.set_title("Input Image (Adverse)", fontsize=14); ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(ref); ax2.set_title("Clear-Weather Reference (RGB)", fontsize=14); ax2.axis('off')

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
            color = CITYSCAPES_PALETTE[cls_id] / 255.0
            patches.append(mpatches.Patch(color=color, label=CITYSCAPES_CLASSES[cls_id]))
        if patches:
            ax_legend.legend(
                handles=patches, loc='center',
                ncol=min(len(patches), 8),
                frameon=False, fontsize='medium', title="Classes Present",
            )

        plt.tight_layout()
        if miou is not None:
            plt.subplots_adjust(top=0.92)
        plt.savefig(self.output_dir / f"result_{idx:03d}.png")
        plt.close()

    def run_inference(self, loader, num_samples=None):
        n = 0
        pbar = tqdm(loader, desc="Inference")
        for batch in pbar:
            pred = self.predict_single_image(batch)
            gt = self.prepare_gt_1024(batch)

            miou = None
            if gt is not None:
                miou = SegMetricsCalculator.compute_image_miou(pred, gt, NUM_CLASSES)
                condition = None
                if 'condition' in batch and isinstance(batch['condition'][0], str):
                    condition = batch['condition'][0]
                self.metrics.update(pred, gt, condition=condition)
                print(f"📊 Image {n:03d} | mIoU: {miou:.4f}")

            self.visualize(batch, pred, gt, idx=n, miou=miou)

            n += 1
            if num_samples is not None and n >= num_samples:
                break

        if n > 0:
            self.metrics.print_report(self.metrics.compute())


if __name__ == "__main__":
    CHECKPOINT_PATH = str(
        _THIS.parent /
        "outputs_weather_sam_mask2former_testv15" /
        "best_E27_mIoU65.68_LR4.0e-05.pth"
    )
    TEST_CSV_PATH = str(
        _THIS.parent.parent / "Datasets" / "acdc_adverse_ref_rgb_val.csv"
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "inference_viz_acdc_v15_E27"

    print(f"Loading v15 model: {Path(CHECKPOINT_PATH).name}")
    model = load_v15_model(CHECKPOINT_PATH, device=DEVICE)

    print(f"Building ACDC val loader: {Path(TEST_CSV_PATH).name}")
    loader = build_acdc_val_loader(TEST_CSV_PATH, batch_size=1, num_workers=4)

    print(f"Output dir: {OUTPUT_DIR}\n")
    runner = InferenceRunner(model, DEVICE, output_dir=OUTPUT_DIR)
    runner.run_inference(loader, num_samples=None)
