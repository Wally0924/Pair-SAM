# weather_dataloader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
import pandas as pd

class WeatherSegmentationDataset(Dataset):
    def __init__(self, csv_file: str, image_size: int = 1024, mode: str = 'train', gps_noise: float = 0.0):
        """
        Args:
            csv_file (str): CSV 路徑
            image_size (int): 統一縮放尺寸 (預設 1024)
            mode (str): 'train', 'val', 或 'test'
            gps_noise (float): GPS 座標的高斯噪聲標準差 (預設 0.0，表示不添加噪聲)
        """
        self.image_size = image_size
        self.mode = mode
        self.gps_noise = gps_noise
        
        # 1. 檢查並讀取 CSV
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ 找不到 CSV 檔案: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        
        # 2. 檢查是否包含特徵路徑欄位 (Feature Caching)
        self.has_cached_features = 'feature_path' in self.data.columns
        if self.has_cached_features:
            print(f"⚡ [Info] Detected 'feature_path' column. Dataset will use cached features when available.")

        # [image-pair] 檢查是否包含 clear-weather embedding 欄位
        self.has_clear_features = 'clear_feature_path' in self.data.columns
        if self.has_clear_features:
            print(f"⚡ [Info] Detected 'clear_feature_path' column. CrossViewAlignment will use clear-weather ViT-H embeddings.")
        
        # 3. 資料清理
        # 支援兩種 ref 欄位名稱：Cityscapes 用 ref_mask_path，ACDC 用 ref_image_path
        self.ref_col = 'ref_mask_path' if 'ref_mask_path' in self.data.columns else 'ref_image_path'

        if mode in ['train', 'val']:
            # 基本檢查：GT 與 Ref 必須存在
            subset = ['gt_path', self.ref_col]
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
        output = {}
        
        # ===========================================================
        # 1. 判斷是否使用 Cache (訓練模式下強制關閉 Cache 以便做增強)
        # ===========================================================
        # 只有當檔案存在時，才使用 cache (現在訓練與驗證都支援 Cache)
        use_cache = False
        if self.has_cached_features and pd.notna(row.get('feature_path')) and os.path.exists(str(row.get('feature_path'))):
            use_cache = True

        # 準備變數 (若是 cache 模式，這些變數可能不會被建立，所以先 init)
        image = None 
        gt_mask = None
        ref_mask = None

        # ===========================================================
        # 2. 載入資料 (Image / Ref Mask / GT)
        # ===========================================================
        
        if use_cache:
            # --- [Cache Mode] ---
            # 讀取預先計算好的特徵
            image_embedding = torch.load(row['feature_path'], weights_only=True)
            output["image_embedding"] = image_embedding
            original_size = (self.image_size, self.image_size) # 假設 cache 都是 1024
            
            # Ref Mask 讀取 (Cache 模式通常不做增強，直接 resize)
            if pd.notna(row.get(self.ref_col)) and os.path.exists(str(row[self.ref_col])):
                ref_mask = cv2.imread(str(row[self.ref_col]))
                ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
            else:
                ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            # GT Mask 讀取
            if pd.notna(row.get('gt_path')) and os.path.exists(str(row['gt_path'])):
                gt_mask = cv2.imread(row['gt_path'], cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = np.full((self.image_size, self.image_size), 255, dtype=np.uint8)
            gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        else:
            # --- [Raw Image Mode] (Training 必走這裡) ---
            
            # A. 讀取原始影像
            image = cv2.imread(row['image_path'])
            if image is None:
                raise ValueError(f"Could not load image: {row['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # B. 讀取 Ref（支援 ref_mask_path / ref_image_path 兩種欄位名稱）
            if pd.notna(row.get(self.ref_col)) and os.path.exists(str(row[self.ref_col])):
                ref_mask = cv2.imread(str(row[self.ref_col]))
                ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
            else:
                ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # C. 讀取 GT Mask
            if pd.notna(row.get('gt_path')) and os.path.exists(str(row['gt_path'])):
                gt_mask = cv2.imread(row['gt_path'], cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = np.full((original_size[0], original_size[1]), 255, dtype=np.uint8)

            # D. 先統一 Resize 到模型輸入尺寸 (例如 1024)
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            # E. 轉 Tensor
            image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
            output["image"] = image_tensor

        # ===========================================================
        # 3. 後續處理 (通用)
        # ===========================================================

        output["original_size"] = original_size

        # [image-pair] 載入 clear-weather ViT-H embedding（作為 CrossViewAlignment 的 f_ref）
        # 若 CSV 有 clear_feature_path 且檔案存在，直接載入預算 embedding (256, 64, 64)
        # 若無，退回全零 tensor（模型仍可訓練，但 CrossViewAlignment 退化為 self-attention）
        if self.has_clear_features and pd.notna(row.get('clear_feature_path')) and os.path.exists(str(row['clear_feature_path'])):
            clear_embedding = torch.load(str(row['clear_feature_path']), weights_only=True)  # (256, 64, 64)
        else:
            clear_embedding = torch.zeros(256, 64, 64, dtype=torch.float32)
        output["clear_embedding"] = clear_embedding

        # 處理 Ref Void Mask (黑色區域檢測)
        ref_mask_tensor = torch.as_tensor(ref_mask).permute(2, 0, 1).float()
        output["reference_mask"] = ref_mask_tensor
        ref_void_mask = (ref_mask_tensor.sum(dim=0) == 0)
        output["ref_void_mask"] = ref_void_mask
        
        # 處理 GT Mask
        output["gt_mask"] = torch.as_tensor(gt_mask).long()

        # 處理 Invalid Mask（optional，目前僅 ACDC 提供）
        # ACDC invalid_mask 實際為二值圖：0 = 有效區域，1 = 無效區域（車頭遮擋等固定盲區）
        if 'invalid_mask' in row and pd.notna(row.get('invalid_mask')) and os.path.exists(str(row['invalid_mask'])):
            inv = cv2.imread(str(row['invalid_mask']), cv2.IMREAD_GRAYSCALE)
            inv = cv2.resize(inv, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            output["invalid_mask"] = torch.as_tensor(inv != 0).bool()  # True = 無效，需排除
        else:
            output["invalid_mask"] = torch.zeros(self.image_size, self.image_size, dtype=torch.bool)

        # 處理 Text Prompts
        active_prompts = []
        # 注意: 這裡判斷 GT 是否全是 0 (背景)
        if gt_mask.max() > 0: 
            unique_classes = np.unique(gt_mask)
            for cls_id in unique_classes:
                if cls_id in self.ID_TO_NAME:
                    active_prompts.append(self.ID_TO_NAME[cls_id])
        
        if not active_prompts:
            active_prompts = ["road"]

        if self.mode == 'train':
            random.shuffle(active_prompts)

        output["text_prompts"] = active_prompts

        # 處理 Location (GPS Noise)
        if 'lat' in row and 'lon' in row:
            lat = float(row['lat'])
            lon = float(row['lon'])
            if self.mode == 'train':
                lat += random.gauss(0, self.gps_noise)
                lon += random.gauss(0, self.gps_noise)
        else:
            lat, lon = 0.0, 0.0

        output["location"] = torch.tensor([lat, lon], dtype=torch.float32)

        # 處理 condition_id（天氣條件索引，ACDC 專用：fog=0, rain=1, snow=2）
        # Cityscapes 等無此欄位的資料集會輸出 -1，模型 forward 中以 -1 作為退回 GPS 路徑的信號
        if 'condition_id' in row and pd.notna(row.get('condition_id')):
            output["condition_id"] = torch.tensor(int(row['condition_id']), dtype=torch.long)
        else:
            output["condition_id"] = torch.tensor(-1, dtype=torch.long)

        return output
    
    @staticmethod
    def collate_fn(batch):
        # 1. 處理通用欄位
        ref_masks = torch.stack([item['reference_mask'] for item in batch])
        gt_masks = torch.stack([item['gt_mask'] for item in batch])

        # 堆疊 ref_void_mask
        ref_void_masks = torch.stack([item['ref_void_mask'] for item in batch])
        invalid_masks = torch.stack([item['invalid_mask'] for item in batch])

        text_prompts = [item['text_prompts'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        locations = torch.stack([item['location'] for item in batch])
        condition_ids = torch.stack([item['condition_id'] for item in batch])

        batch_dict = {
            "reference_mask": ref_masks,
            "ref_void_mask": ref_void_masks,
            "gt_mask": gt_masks,
            "invalid_mask": invalid_masks,
            "text_prompts": text_prompts,
            "original_size": original_sizes,
            "location": locations,
            "condition_id": condition_ids,
        }

        # [image-pair] clear-weather ViT-H embedding
        batch_dict['clear_embedding'] = torch.stack([item['clear_embedding'] for item in batch])

        # 2. 動態處理影像輸入
        if 'image_embedding' in batch[0]:
            batch_dict['image_embedding'] = torch.stack([item['image_embedding'] for item in batch])
        elif 'image' in batch[0]:
            batch_dict['image'] = torch.stack([item['image'] for item in batch])

        return batch_dict
