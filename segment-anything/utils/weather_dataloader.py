import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
import pandas as pd

class WeatherSegmentationDataset(Dataset):
    def __init__(self, csv_file: str, image_size: int = 1024, mode: str = 'train'):
        """
        Args:
            csv_file (str): CSV 路徑
            image_size (int): 統一縮放尺寸 (預設 1024)
            mode (str): 'train', 'val', 或 'test'
        """
        self.image_size = image_size
        self.mode = mode
        
        # 1. 檢查並讀取 CSV
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ 找不到 CSV 檔案: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        
        # 2. 檢查是否包含特徵路徑欄位 (Feature Caching)
        self.has_cached_features = 'feature_path' in self.data.columns
        if self.has_cached_features:
            print(f"⚡ [Info] Detected 'feature_path' column. Dataset will use cached features when available.")
        
        # 3. 資料清理
        if mode in ['train', 'val']:
            # 基本檢查：GT 與 Ref Mask 必須存在
            subset = ['gt_path', 'ref_mask_path']
            # 如果是快取模式，也要檢查 feature_path 是否存在 (非 NaN)
            if self.has_cached_features:
                subset.append('feature_path')
                
            initial_len = len(self.data)
            self.data = self.data.dropna(subset=subset)
            if len(self.data) < initial_len:
                print(f"⚠️ Warning: Removed {initial_len - len(self.data)} invalid samples.")
        
        print(f"[{mode.upper()}] Dataset loaded from {csv_file}. Total samples: {len(self.data)}")

        # 定義類別 ID 對應
        self.CLASS_MAP = {
            "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
            "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
            "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
            "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
        }
        self.ID_TO_NAME = {v: k for k, v in self.CLASS_MAP.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 輸出字典
        output = {}
        
        # -----------------------------------------------------------
        # 1. 讀取影像或特徵 (Image or Feature)
        # -----------------------------------------------------------
        # 判斷是否使用快取特徵
        use_cache = False
        if self.has_cached_features and pd.notna(row['feature_path']) and os.path.exists(row['feature_path']):
            use_cache = True

        if use_cache:
            # === 快取模式 (極速) ===
            # 直接載入 .pt 檔 (Shape: 256, 64, 64)
            # 注意：這裡載入到 CPU，由 DataLoader worker 處理
            image_embedding = torch.load(row['feature_path'])
            output["image_embedding"] = image_embedding
            
            # 在快取模式下，我們通常沒有讀取原圖，所以 original_size 設為模型輸入尺寸
            # 若訓練需要嚴格的原始尺寸，需在 precompute 階段存入 CSV
            original_size = (self.image_size, self.image_size)
            
        else:
            # === 原始模式 (讀圖) ===
            image = cv2.imread(row['image_path'])
            if image is None:
                raise ValueError(f"Could not load image: {row['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # Resize & Preprocess
            image = cv2.resize(image, (self.image_size, self.image_size))
            image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
            output["image"] = image_tensor

        output["original_size"] = original_size

        # -----------------------------------------------------------
        # 2. 讀取參考遮罩 (Reference Mask)
        # -----------------------------------------------------------
        # 無論是否用快取，MaskEncoder 還是需要讀取參考圖
        ref_mask_path = row.get('ref_mask_path', None)
        if pd.notna(ref_mask_path) and os.path.exists(str(ref_mask_path)):
            ref_mask = cv2.imread(ref_mask_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        else:
            ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
        ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        output["reference_mask"] = torch.as_tensor(ref_mask).permute(2, 0, 1).float()

        # -----------------------------------------------------------
        # 3. 讀取 Ground Truth
        # -----------------------------------------------------------
        gt_path = row.get('gt_path', None)
        has_gt = False
        if pd.notna(gt_path) and os.path.exists(str(gt_path)):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            has_gt = True
        else:
            gt_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Resize GT (必須與模型輸出對齊)
        gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        output["gt_mask"] = torch.as_tensor(gt_mask).long()

        # -----------------------------------------------------------
        # 4. 生成 Text Prompts
        # -----------------------------------------------------------
        active_prompts = []
        if has_gt:
            unique_classes = np.unique(gt_mask)
            for cls_id in unique_classes:
                if cls_id in self.ID_TO_NAME:
                    active_prompts.append(self.ID_TO_NAME[cls_id])
        else:
            active_prompts = ["road"]

        if not active_prompts:
            active_prompts = ["road"]

        # 訓練時限制 Prompt 數量
        if self.mode == 'train' and len(active_prompts) > 3:
            active_prompts = random.sample(active_prompts, 3)
            
        output["text_prompts"] = active_prompts

        return output
    
    @staticmethod
    def collate_fn(batch):
        """
        修正後的 Collate Function:
        自動偵測是堆疊 'image' 還是 'image_embedding'
        """
        # 1. 處理通用欄位
        ref_masks = torch.stack([item['reference_mask'] for item in batch])
        gt_masks = torch.stack([item['gt_mask'] for item in batch])
        text_prompts = [item['text_prompts'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        batch_dict = {
            "reference_mask": ref_masks,
            "gt_mask": gt_masks,
            "text_prompts": text_prompts,
            "original_size": original_sizes
        }

        # 2. 動態處理影像輸入
        # 檢查第一個樣本是用 embedding 還是 raw image
        if 'image_embedding' in batch[0]:
            batch_dict['image_embedding'] = torch.stack([item['image_embedding'] for item in batch])
        elif 'image' in batch[0]:
            batch_dict['image'] = torch.stack([item['image'] for item in batch])
            
        return batch_dict