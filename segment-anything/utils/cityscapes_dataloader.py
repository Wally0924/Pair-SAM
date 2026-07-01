# cityscapes_dataloader.py
"""
Stage-1 Cityscapes 語意分割 Dataset（WeatherSAM encoder pretrain 專用）。

用途：
    以 Cityscapes 晴天影像對「乾淨版 WeatherSAM」(use_vgg_adapter=False, cond=False)
    做全量 fine-tune，取得域內化的 ViT-H encoder。此 Dataset 刻意與 ACDC 用的
    WeatherSegmentationDataset 解耦——不需要 clear reference、condition_id、
    invalid_mask、feature cache 等惡劣天氣專屬欄位，只提供影像 + 19 類 trainId 標註。

輸出格式（與 model.preprocess / model.forward 的期望一致）：
    'image': (3, H, W) float，值域 0~255（正規化與 pad 由 model.preprocess 處理）
    'label': (H, W)   long，值域 0~18，255 = ignore

train 模式：random scale → random crop(crop_size²) → h-flip → 輕度光度抖動
val   模式：回傳完整 2048×1024 影像與標註；tiling 推論由訓練腳本負責

檔名配對（Cityscapes 慣例，影像與標註共用同一 frame 前綴）：
    Images/leftImg8bit/<split>/<city>/<prefix>_leftImg8bit.png
    GT/gtFine/<split>/<city>/<prefix>_gtFine_labelTrainIds.png
"""
import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Cityscapes 19 類名稱（trainId 0~18），順序與 WeatherSAM.CLASS_MAP 一致
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

IGNORE_INDEX = 255


class CityscapesPretrainDataset(Dataset):
    def __init__(
        self,
        images_root: str,
        gt_root: str,
        split: str = "train",
        crop_size: int = 1024,
        scale_range: tuple = (0.5, 2.0),
    ):
        """
        Args:
            images_root: 例如 /home/rvl1421/Datasets/Cityscapes/Images/leftImg8bit
            gt_root:     例如 /home/rvl1421/Datasets/Cityscapes/GT/gtFine
            split:       'train' / 'val'
            crop_size:   train 模式的方形裁切邊長（ViT-H 輸入為 1024）
            scale_range: train 模式 random scale 範圍
        """
        self.split = split
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.is_train = (split == "train")

        img_dir = os.path.join(images_root, split)
        gt_dir = os.path.join(gt_root, split)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"❌ 找不到影像目錄: {img_dir}")

        # 掃描所有 leftImg8bit，並推導對應的 labelTrainIds 路徑
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*", "*_leftImg8bit.png")))
        self.samples = []
        missing = 0
        for img_path in image_paths:
            city = os.path.basename(os.path.dirname(img_path))
            base = os.path.basename(img_path)[: -len("_leftImg8bit.png")]
            label_path = os.path.join(gt_dir, city, base + "_gtFine_labelTrainIds.png")
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))
            else:
                missing += 1

        if not self.samples:
            raise RuntimeError(
                f"❌ 在 {img_dir} 找不到任何有效的 (影像, labelTrainIds) 配對。"
                f"請確認 GT 目錄 {gt_dir} 內含 *_gtFine_labelTrainIds.png。"
            )
        print(f"[Cityscapes/{split}] {len(self.samples)} samples"
              + (f"（{missing} 張缺少 labelTrainIds，已略過）" if missing else ""))

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # 增強（僅 train）
    # ------------------------------------------------------------------
    def _random_scale(self, image: np.ndarray, label: np.ndarray):
        s = random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        nh, nw = int(round(h * s)), int(round(w * s))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return image, label

    def _pad_to_crop(self, image: np.ndarray, label: np.ndarray):
        """若縮放後小於 crop_size，補邊（影像補 0，標註補 255 ignore）。"""
        h, w = image.shape[:2]
        pad_h = max(0, self.crop_size - h)
        pad_w = max(0, self.crop_size - w)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
        return image, label

    def _random_crop(self, image: np.ndarray, label: np.ndarray):
        h, w = image.shape[:2]
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        image = image[top:top + self.crop_size, left:left + self.crop_size]
        label = label[top:top + self.crop_size, left:left + self.crop_size]
        return image, label

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        """輕度亮度/對比抖動（在 uint8 值域操作後 clip）。"""
        brightness = random.uniform(-15, 15)
        contrast = random.uniform(0.9, 1.1)
        image = image.astype(np.float32) * contrast + brightness
        return np.clip(image, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"無法讀取影像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # labelTrainIds 為單通道，值 0~18 與 255；務必以 UNCHANGED/GRAYSCALE 讀取避免被轉成 3 通道
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"無法讀取標註: {label_path}")

        if self.is_train:
            image, label = self._random_scale(image, label)
            image, label = self._pad_to_crop(image, label)
            image, label = self._random_crop(image, label)
            if random.random() < 0.5:  # 水平翻轉
                image = image[:, ::-1, :].copy()
                label = label[:, ::-1].copy()
            image = self._photometric(image)
        # val：保留原始 2048×1024，tiling 由訓練腳本處理

        image_tensor = torch.as_tensor(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        label_tensor = torch.as_tensor(np.ascontiguousarray(label)).long()
        return {"image": image_tensor, "label": label_tensor}
