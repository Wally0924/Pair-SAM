"""
seg_metrics.py
語意分割評估指標計算器（Cityscapes 19-class 標準）

支援指標：
  - mIoU        (全域累積，Cityscapes 標準)
  - per-class IoU
  - per-condition mIoU  (ACDC: fog / rain / snow / night)
  - mAcc        (mean Accuracy = per-class recall 平均)

使用方式：
    calculator = SegMetricsCalculator(classes=CLASSES)
    calculator.update(pred_mask, gt_mask, condition="fog")   # 每張影像呼叫一次
    results = calculator.compute()
    calculator.print_report(results)
"""

import numpy as np
from typing import Optional

ACDC_CONDITIONS = ["fog", "rain", "snow", "night"]

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]


class SegMetricsCalculator:
    """
    累積式語意分割指標計算器。
    每張影像呼叫 update()，全部跑完後呼叫 compute() 取得結果。
    """

    def __init__(self, classes: list = CITYSCAPES_CLASSES, ignore_index: int = 255):
        self.classes = classes
        self.num_classes = len(classes)
        self.ignore_index = ignore_index

        # 全域累積（用於 mIoU / per-class IoU）
        self.total_intersection = np.zeros(self.num_classes, dtype=np.int64)
        self.total_union        = np.zeros(self.num_classes, dtype=np.int64)

        # mAcc 累積：TP 與 GT 像素總數（per class）
        self.total_tp           = np.zeros(self.num_classes, dtype=np.int64)
        self.total_gt_pixels    = np.zeros(self.num_classes, dtype=np.int64)

        # per-condition 累積（ACDC）
        self.condition_intersection = {c: np.zeros(self.num_classes, dtype=np.int64) for c in ACDC_CONDITIONS}
        self.condition_union        = {c: np.zeros(self.num_classes, dtype=np.int64) for c in ACDC_CONDITIONS}

        self.num_images = 0

    def reset(self):
        self.__init__(self.classes, self.ignore_index)

    def update(
        self,
        pred_mask: np.ndarray,
        gt_mask:   np.ndarray,
        condition: Optional[str] = None,
    ):
        """
        累積單張影像的統計量。

        Args:
            pred_mask:  (H, W) int，值域 0 ~ num_classes-1
            gt_mask:    (H, W) int，ignore_index 區域會被排除
            condition:  ACDC 天氣條件字串（"fog"/"rain"/"snow"/"night"），
                        非 ACDC 資料集傳 None
        """
        assert pred_mask.shape == gt_mask.shape, "pred 與 gt 尺寸不符"

        valid = gt_mask != self.ignore_index  # (H, W) bool

        for cls_id in range(self.num_classes):
            pred_cls = pred_mask == cls_id
            gt_cls   = gt_mask   == cls_id

            tp    = np.logical_and(pred_cls, gt_cls)[valid].sum()
            fp    = np.logical_and(pred_cls, ~gt_cls)[valid].sum()
            fn    = np.logical_and(~pred_cls, gt_cls)[valid].sum()
            union = tp + fp + fn

            self.total_intersection[cls_id] += tp
            self.total_union[cls_id]        += union
            self.total_tp[cls_id]           += tp
            self.total_gt_pixels[cls_id]    += (gt_cls & valid).sum()

            if condition in ACDC_CONDITIONS:
                self.condition_intersection[condition][cls_id] += tp
                self.condition_union[condition][cls_id]        += union

        self.num_images += 1

    # ------------------------------------------------------------------
    # 計算函式
    # ------------------------------------------------------------------

    def _iou_from_accum(self, intersection: np.ndarray, union: np.ndarray) -> dict:
        """從累積的 intersection / union 計算 per-class IoU 與 mIoU。"""
        per_class = {}
        valid_ious = []
        for cls_id in range(self.num_classes):
            if union[cls_id] > 0:
                iou = float(intersection[cls_id]) / float(union[cls_id])
                per_class[self.classes[cls_id]] = iou
                valid_ious.append(iou)
            else:
                per_class[self.classes[cls_id]] = float("nan")
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        return {"miou": miou, "per_class": per_class}

    def compute(self) -> dict:
        """
        計算所有指標，回傳 dict。

        Returns:
            {
                "miou":              float,
                "per_class_iou":     dict[class_name -> float | nan],
                "macc":              float,
                "per_class_acc":     dict[class_name -> float | nan],
                "per_condition_miou": dict[condition -> float],   # 僅含有資料的條件
                "num_images":        int,
            }
        """
        # ── mIoU & per-class IoU ──
        global_result = self._iou_from_accum(self.total_intersection, self.total_union)

        # ── mAcc ──
        per_class_acc = {}
        valid_accs = []
        for cls_id in range(self.num_classes):
            if self.total_gt_pixels[cls_id] > 0:
                acc = float(self.total_tp[cls_id]) / float(self.total_gt_pixels[cls_id])
                per_class_acc[self.classes[cls_id]] = acc
                valid_accs.append(acc)
            else:
                per_class_acc[self.classes[cls_id]] = float("nan")
        macc = float(np.mean(valid_accs)) if valid_accs else 0.0

        # ── per-condition mIoU ──
        per_condition_miou = {}
        for cond in ACDC_CONDITIONS:
            if self.condition_union[cond].sum() > 0:
                result = self._iou_from_accum(
                    self.condition_intersection[cond],
                    self.condition_union[cond]
                )
                per_condition_miou[cond] = result["miou"]

        return {
            "miou":               global_result["miou"],
            "per_class_iou":      global_result["per_class"],
            "macc":               macc,
            "per_class_acc":      per_class_acc,
            "per_condition_miou": per_condition_miou,
            "num_images":         self.num_images,
        }

    # ------------------------------------------------------------------
    # 報告輸出
    # ------------------------------------------------------------------

    @staticmethod
    def compute_image_miou(
        pred_mask: np.ndarray,
        gt_mask:   np.ndarray,
        num_classes: int = 19,
        ignore_index: int = 255,
    ) -> float:
        """單張影像的 mIoU（不影響全域累積器）。"""
        valid = gt_mask != ignore_index
        ious = []
        for cls_id in range(num_classes):
            pred_cls = pred_mask == cls_id
            gt_cls   = gt_mask   == cls_id
            tp    = np.logical_and(pred_cls,  gt_cls)[valid].sum()
            fp    = np.logical_and(pred_cls, ~gt_cls)[valid].sum()
            fn    = np.logical_and(~pred_cls, gt_cls)[valid].sum()
            union = tp + fp + fn
            if union > 0:
                ious.append(float(tp) / float(union))
        return float(np.mean(ious)) if ious else 0.0

    def print_report(self, results: dict):
        sep  = "=" * 62
        dash = "-" * 62

        print(f"\n{sep}")
        print(f"  Segmentation Evaluation Report  ({results['num_images']} images)")
        print(sep)

        # per-class IoU & Acc
        print(f"\n{'Class':<22} {'IoU':>10} {'Acc(Recall)':>14}")
        print(dash)
        for cls_name in self.classes:
            iou = results["per_class_iou"][cls_name]
            acc = results["per_class_acc"][cls_name]
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
            acc_str = f"{acc:.4f}" if not np.isnan(acc) else "      N/A"
            print(f"{cls_name:<22} {iou_str:>10} {acc_str:>14}")

        print(dash)
        print(f"{'mIoU':<22} {results['miou']:>10.4f}")
        print(f"{'mAcc':<22} {results['macc']:>10.4f}")

        # per-condition mIoU
        if results["per_condition_miou"]:
            print(f"\n{sep}")
            print("  Per-Condition mIoU (ACDC)")
            print(dash)
            for cond, miou in results["per_condition_miou"].items():
                print(f"  {cond:<10}  mIoU = {miou:.4f}")

        print(f"{sep}\n")
