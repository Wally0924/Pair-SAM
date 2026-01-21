import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
import pandas as pd  # 新增: 用於讀取 CSV

class WeatherSegmentationDataset(Dataset):
    def __init__(self, csv_file: str, image_size: int = 1024, mode: str = 'train'):
        """
        Args:
            csv_file (str): 對應表 CSV 的路徑 (例如 'data/weather_dataset/train.csv')
            image_size (int): 統一縮放尺寸 (SAM 預設 1024)
            mode (str): 'train', 'val', 或 'test'
        """
        self.image_size = image_size
        self.mode = mode
        
        # 1. 檢查並讀取 CSV
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ 找不到 CSV 檔案: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        
        # 2. 資料清理 (針對 train/val 模式)
        # 確保訓練時 GT 和 Ref Mask 的路徑都存在，避免訓練中斷
        if mode in ['train', 'val']:
            # 濾掉任何路徑為空的資料 (雖然驗證腳本已經檢查過，但多一層保護比較好)
            initial_len = len(self.data)
            self.data = self.data.dropna(subset=['gt_path', 'ref_mask_path'])
            if len(self.data) < initial_len:
                print(f"⚠️ Warning: Removed {initial_len - len(self.data)} invalid samples from {mode} set.")
        
        print(f"[{mode.upper()}] Dataset loaded from {csv_file}. Total samples: {len(self.data)}")

        # 定義類別 ID 對應 (保持不變)
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
        # 取得 CSV 中的該行資料
        row = self.data.iloc[idx]
        
        # -----------------------------------------------------------
        # 1. 讀取惡劣天氣影像 (Bad Image)
        # -----------------------------------------------------------
        image = cv2.imread(row['image_path'])
        if image is None:
            # 防呆：如果路徑對但讀不到圖 (極少見)
            raise ValueError(f"Could not load image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # -----------------------------------------------------------
        # 2. 讀取參考遮罩 (Reference Mask)
        # -----------------------------------------------------------
        ref_mask_path = row.get('ref_mask_path', None)
        
        if pd.notna(ref_mask_path) and os.path.exists(str(ref_mask_path)):
            ref_mask = cv2.imread(ref_mask_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        else:
            # 如果是 Test set 或者缺檔，給全黑圖
            ref_mask = np.zeros_like(image)

        # -----------------------------------------------------------
        # 3. 讀取 Ground Truth Label (類別索引圖)
        # -----------------------------------------------------------
        gt_path = row.get('gt_path', None)
        has_gt = False
        
        if pd.notna(gt_path) and os.path.exists(str(gt_path)):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            has_gt = True
        else:
            # 如果是 Test set，給一個空的 mask (全 0)
            # 注意: 這裡給 np.uint8 格式，尺寸先跟原圖一樣
            gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # -----------------------------------------------------------
        # Resize & Preprocess
        # -----------------------------------------------------------
        original_size = image.shape[:2]
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        # 使用 INTER_NEAREST 避免 mask 邊緣產生不存在的顏色/類別
        ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Transform to Tensor
        image_tensor = torch.as_tensor(image).permute(2, 0, 1).float() 
        ref_mask_tensor = torch.as_tensor(ref_mask).permute(2, 0, 1).float()
        gt_tensor = torch.as_tensor(gt_mask).long()

        # -----------------------------------------------------------
        # 生成 Text Prompts
        # -----------------------------------------------------------
        active_prompts = []
        
        if has_gt:
            # 訓練/驗證模式：從 GT 裡找出存在的類別
            unique_classes = np.unique(gt_mask)
            for cls_id in unique_classes:
                if cls_id in self.ID_TO_NAME:
                    active_prompts.append(self.ID_TO_NAME[cls_id])
        else:
            # 測試模式 (無 GT)：給一個預設 Prompt 列表
            # 這裡可以給 "road" 或者是全部類別，視推論策略而定
            active_prompts = ["road", "car", "building"] 

        # 防呆：如果是空的
        if not active_prompts:
            active_prompts = ["road"]

        # 訓練時隨機採樣，避免 Prompt 太多導致記憶體爆炸
        if self.mode == 'train' and len(active_prompts) > 3:
            active_prompts = random.sample(active_prompts, 3)

        return {
            "image": image_tensor,
            "reference_mask": ref_mask_tensor,
            "gt_mask": gt_tensor,
            "text_prompts": active_prompts,
            "original_size": original_size
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        保持不變
        """
        images = torch.stack([item['image'] for item in batch])
        ref_masks = torch.stack([item['reference_mask'] for item in batch])
        gt_masks = torch.stack([item['gt_mask'] for item in batch])
        text_prompts = [item['text_prompts'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        return {
            "image": images,
            "reference_mask": ref_masks,
            "gt_mask": gt_masks,
            "text_prompts": text_prompts,
            "original_size": original_sizes
        }