# -*- coding: utf-8 -*-
"""產生 Cityscapes 的 PairSegmentationDataset CSV(m2f CS 預訓練用)。

一次性腳本:掃描 Cityscapes leftImg8bit 影像,配對 gtFine labelTrainIds,
輸出與 acdc_adverse_ref_rgb_{train,val}.csv 同欄位格式的 CSV。

欄位語意(對齊 ACDC CSV,零 dataloader 改動):
  - ref_image_path = image_path 自身(self-reference)。CS 預訓練關閉
    adapter(--no-use_vgg_adapter),forward 不會用到 clear_image,
    但 dataset 一定讀 ref 欄位,故指向自身避免 dropna 剔除與 imread 失敗。
  - condition='normal', condition_id=0:dataset 為 strict mode(id 必須 0-3),
    借用 index 0;CS 階段條件 token 等效「與條件無關的可學習常數」。
  - invalid_mask 留空:dataset 自動補全零遮罩(全有效)。
"""
import csv
import os
import sys

CS_ROOT = "/home/rvl1421/Datasets/Cityscapes"
IMG_ROOT = os.path.join(CS_ROOT, "Images", "leftImg8bit")
GT_ROOT = os.path.join(CS_ROOT, "GT", "gtFine")
OUT_DIR = "/home/rvl1421/SAM_research-1/Datasets"

FIELDS = ["image_path", "ref_image_path", "gt_path", "condition", "condition_id", "invalid_mask"]


def build_split(split: str) -> str:
    rows, missing = [], 0
    split_dir = os.path.join(IMG_ROOT, split)
    for city in sorted(os.listdir(split_dir)):
        city_dir = os.path.join(split_dir, city)
        for fname in sorted(os.listdir(city_dir)):
            if not fname.endswith("_leftImg8bit.png"):
                continue
            img_path = os.path.join(city_dir, fname)
            gt_path = os.path.join(
                GT_ROOT, split, city,
                fname.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png"))
            if not os.path.isfile(gt_path):
                missing += 1
                continue
            rows.append({
                "image_path": img_path,
                "ref_image_path": img_path,   # self-reference(見檔頭說明)
                "gt_path": gt_path,
                "condition": "normal",
                "condition_id": 0,
                "invalid_mask": "",
            })
    out_csv = os.path.join(OUT_DIR, f"cityscapes_m2f_{split}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{split}] 寫入 {len(rows)} 筆 → {out_csv}" + (f"(缺 GT 跳過 {missing})" if missing else ""))
    return out_csv


if __name__ == "__main__":
    for split in ("train", "val"):
        build_split(split)
    # 驗證筆數符合 Cityscapes 官方 split
    expected = {"train": 2975, "val": 500}
    for split, n_exp in expected.items():
        out_csv = os.path.join(OUT_DIR, f"cityscapes_m2f_{split}.csv")
        with open(out_csv) as f:
            n = sum(1 for _ in f) - 1
        status = "✓" if n == n_exp else f"✗ 預期 {n_exp}"
        print(f"  {split}: {n} 筆 {status}")
        if n != n_exp:
            sys.exit(1)
