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
        output = {}
        
        # -----------------------------------------------------------
        # 1. 讀取影像或特徵 (Image or Feature)
        # -----------------------------------------------------------
        use_cache = False
        if self.has_cached_features and pd.notna(row['feature_path']) and os.path.exists(row['feature_path']):
            use_cache = True

        if use_cache:
            # === 快取模式 ===
            image_embedding = torch.load(row['feature_path'])
            output["image_embedding"] = image_embedding
            original_size = (self.image_size, self.image_size)
        else:
            # === 原始模式 ===
            image = cv2.imread(row['image_path'])
            if image is None:
                raise ValueError(f"Could not load image: {row['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # Resize
            image = cv2.resize(image, (self.image_size, self.image_size))
            image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
            output["image"] = image_tensor

        output["original_size"] = original_size

        # -----------------------------------------------------------
        # 2. 讀取參考遮罩 (Reference Mask) & 製作 Void Mask
        # -----------------------------------------------------------
        ref_mask_path = row.get('ref_mask_path', None)
        if pd.notna(ref_mask_path) and os.path.exists(str(ref_mask_path)):
            ref_mask = cv2.imread(ref_mask_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        else:
            ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
        ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # 轉為 Tensor: (3, H, W)
        ref_mask_tensor = torch.as_tensor(ref_mask).permute(2, 0, 1).float()
        output["reference_mask"] = ref_mask_tensor
        
        # [保留] 檢測黑色區域 (Void/Black Detection)
        # 邏輯：如果在 RGB 三個通道上的總和為 0，代表是全黑 (0,0,0)
        ref_void_mask = (ref_mask_tensor.sum(dim=0) == 0) 
        output["ref_void_mask"] = ref_void_mask

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

        gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        output["gt_mask"] = torch.as_tensor(gt_mask).long()

        # -----------------------------------------------------------
        # 4. 生成 Text Prompts (回復為全類別訓練)
        # -----------------------------------------------------------
        active_prompts = []
        if has_gt:
            # 取得 GT 中存在的所有唯一類別 ID
            unique_classes = np.unique(gt_mask)
            for cls_id in unique_classes:
                # 確保 ID 在我們定義的 19 類中 (過濾掉 255 或其他無效 ID)
                if cls_id in self.ID_TO_NAME:
                    active_prompts.append(self.ID_TO_NAME[cls_id])
        
        # 預設值 (防空)
        if not active_prompts:
            active_prompts = ["road"]

        # [修改] 訓練模式下，不再丟棄任何類別，但進行洗牌
        if self.mode == 'train':
            # 隨機打亂順序，避免模型記住 "Road 總是排第一個"
            random.shuffle(active_prompts)
            
            # 若您之前有設定 Prompt 數量上限 (如最多 3 個)，這裡已經移除了，
            # 現在會回傳 GT 裡有的 "所有" 類別。

        output["text_prompts"] = active_prompts

        # -----------------------------------------------------------
        # 5. 讀取 GPS 座標 (Location)
        # -----------------------------------------------------------
        if 'lat' in row and 'lon' in row:
            lat = float(row['lat'])
            lon = float(row['lon'])
            if self.mode == 'train':
                # 添加高斯噪聲
                lat += random.gauss(0, self.gps_noise)
                lon += random.gauss(0, self.gps_noise)
        else:
            # 防呆機制：如果沒有 GPS，給定一個預設值 (例如 0,0) 或報錯
            # 建議訓練前檢查 CSV 完整性
            lat, lon = 0.0, 0.0

        # 轉為 Tensor (2,)
        output["location"] = torch.tensor([lat, lon], dtype=torch.float32)

        return output
    
    @staticmethod
    def collate_fn(batch):
        # 1. 處理通用欄位
        ref_masks = torch.stack([item['reference_mask'] for item in batch])
        gt_masks = torch.stack([item['gt_mask'] for item in batch])

        # 堆疊 ref_void_mask
        ref_void_masks = torch.stack([item['ref_void_mask'] for item in batch])

        text_prompts = [item['text_prompts'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        locations = torch.stack([item['location'] for item in batch])
        
        batch_dict = {
            "reference_mask": ref_masks,
            "ref_void_mask": ref_void_masks,
            "gt_mask": gt_masks,
            "text_prompts": text_prompts,
            "original_size": original_sizes,
            "location": locations
        }

        # 2. 動態處理影像輸入
        if 'image_embedding' in batch[0]:
            batch_dict['image_embedding'] = torch.stack([item['image_embedding'] for item in batch])
        elif 'image' in batch[0]:
            batch_dict['image'] = torch.stack([item['image'] for item in batch])
            
        return batch_dict