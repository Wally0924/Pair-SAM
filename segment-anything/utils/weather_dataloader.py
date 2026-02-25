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

    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]
    #     output = {}
        
    #     # -----------------------------------------------------------
    #     # 1. 讀取影像或特徵 (Image or Feature)
    #     # -----------------------------------------------------------
    #     use_cache = False
    #     if self.has_cached_features and pd.notna(row['feature_path']) and os.path.exists(row['feature_path']):
    #         use_cache = True

    #     if use_cache:
    #         # === 快取模式 ===
    #         image_embedding = torch.load(row['feature_path'])
    #         output["image_embedding"] = image_embedding
    #         original_size = (self.image_size, self.image_size)
    #     else:
    #         # === 原始模式 ===
    #         image = cv2.imread(row['image_path'])
    #         if image is None:
    #             raise ValueError(f"Could not load image: {row['image_path']}")
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         original_size = image.shape[:2]
            
    #         # Resize
    #         # image = cv2.resize(image, (self.image_size, self.image_size))
    #         image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
    #         image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
    #         output["image"] = image_tensor

    #     output["original_size"] = original_size

    #     # -----------------------------------------------------------
    #     # 2. 讀取參考遮罩 (Reference Mask) & 製作 Void Mask
    #     # -----------------------------------------------------------
    #     ref_mask_path = row.get('ref_mask_path', None)
    #     if pd.notna(ref_mask_path) and os.path.exists(str(ref_mask_path)):
    #         ref_mask = cv2.imread(ref_mask_path)
    #         ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
    #     else:
    #         ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
    #     # ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
    #     ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
    #     # 轉為 Tensor: (3, H, W)
    #     ref_mask_tensor = torch.as_tensor(ref_mask).permute(2, 0, 1).float()
    #     output["reference_mask"] = ref_mask_tensor
        
    #     # [保留] 檢測黑色區域 (Void/Black Detection)
    #     # 邏輯：如果在 RGB 三個通道上的總和為 0，代表是全黑 (0,0,0)
    #     ref_void_mask = (ref_mask_tensor.sum(dim=0) == 0) 
    #     output["ref_void_mask"] = ref_void_mask

    #     # -----------------------------------------------------------
    #     # 3. 讀取 Ground Truth
    #     # -----------------------------------------------------------
    #     gt_path = row.get('gt_path', None)
    #     has_gt = False
    #     if pd.notna(gt_path) and os.path.exists(str(gt_path)):
    #         gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    #         has_gt = True
    #     else:
    #         gt_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

    #     gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
    #     output["gt_mask"] = torch.as_tensor(gt_mask).long()

    #     # -----------------------------------------------------------
    #     # 4. 生成 Text Prompts (回復為全類別訓練)
    #     # -----------------------------------------------------------
    #     active_prompts = []
    #     if has_gt:
    #         # 取得 GT 中存在的所有唯一類別 ID
    #         unique_classes = np.unique(gt_mask)
    #         for cls_id in unique_classes:
    #             # 確保 ID 在我們定義的 19 類中 (過濾掉 255 或其他無效 ID)
    #             if cls_id in self.ID_TO_NAME:
    #                 active_prompts.append(self.ID_TO_NAME[cls_id])
        
    #     # 預設值 (防空)
    #     if not active_prompts:
    #         active_prompts = ["road"]

    #     # [修改] 訓練模式下，不再丟棄任何類別，但進行洗牌
    #     if self.mode == 'train':
    #         # 隨機打亂順序，避免模型記住 "Road 總是排第一個"
    #         random.shuffle(active_prompts)
            
    #         # 若您之前有設定 Prompt 數量上限 (如最多 3 個)，這裡已經移除了，
    #         # 現在會回傳 GT 裡有的 "所有" 類別。

    #     output["text_prompts"] = active_prompts

    #     # -----------------------------------------------------------
    #     # 5. 讀取 GPS 座標 (Location)
    #     # -----------------------------------------------------------
    #     if 'lat' in row and 'lon' in row:
    #         lat = float(row['lat'])
    #         lon = float(row['lon'])
    #         if self.mode == 'train':
    #             # 添加高斯噪聲
    #             lat += random.gauss(0, self.gps_noise)
    #             lon += random.gauss(0, self.gps_noise)
    #     else:
    #         # 防呆機制：如果沒有 GPS，給定一個預設值 (例如 0,0) 或報錯
    #         # 建議訓練前檢查 CSV 完整性
    #         lat, lon = 0.0, 0.0

    #     # 轉為 Tensor (2,)
    #     output["location"] = torch.tensor([lat, lon], dtype=torch.float32)

    #     return output
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        output = {}
        
        # ===========================================================
        # 1. 判斷是否使用 Cache (訓練模式下強制關閉 Cache 以便做增強)
        # ===========================================================
        use_cache = False
        # 只有在非訓練模式 (val/test) 且檔案存在時，才使用 cache
        if self.mode != 'train' and self.has_cached_features and pd.notna(row['feature_path']) and os.path.exists(row['feature_path']):
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
            image_embedding = torch.load(row['feature_path'])
            output["image_embedding"] = image_embedding
            original_size = (self.image_size, self.image_size) # 假設 cache 都是 1024
            
            # Ref Mask 讀取 (Cache 模式通常不做增強，直接 resize)
            if pd.notna(row.get('ref_mask_path')) and os.path.exists(str(row['ref_mask_path'])):
                ref_mask = cv2.imread(row['ref_mask_path'])
                ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
            else:
                ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)

            # GT Mask 讀取
            if pd.notna(row.get('gt_path')) and os.path.exists(str(row['gt_path'])):
                gt_mask = cv2.imread(row['gt_path'], cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        else:
            # --- [Raw Image Mode] (Training 必走這裡) ---
            
            # A. 讀取原始影像
            image = cv2.imread(row['image_path'])
            if image is None:
                raise ValueError(f"Could not load image: {row['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # B. 讀取 Ref Mask
            if pd.notna(row.get('ref_mask_path')) and os.path.exists(str(row['ref_mask_path'])):
                ref_mask = cv2.imread(row['ref_mask_path'])
                ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
            else:
                ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # C. 讀取 GT Mask
            if pd.notna(row.get('gt_path')) and os.path.exists(str(row['gt_path'])):
                gt_mask = cv2.imread(row['gt_path'], cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)

            # D. 先統一 Resize 到模型輸入尺寸 (例如 1024)
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            # ===========================================================
            # 🔥 [關鍵修改] Data Augmentation 資料增強
            # ===========================================================
            if self.mode == 'train':
                # 1. 隨機水平翻轉 (Random Horizontal Flip) - 機率 50%
                if random.random() > 0.5:
                    image = cv2.flip(image, 1)
                    ref_mask = cv2.flip(ref_mask, 1)
                    gt_mask = cv2.flip(gt_mask, 1)
                
                # # 2. 隨機色彩擾動 (Color Jitter) - 調整亮度與對比度
                # # 幫助模型適應不同天氣的光線變化
                # if random.random() > 0.5: 
                #     # 亮度 (Brightness): -30 ~ +30
                #     brightness = random.randint(-30, 30)
                #     # 對比度 (Contrast): 0.8 ~ 1.2
                #     contrast = random.uniform(0.8, 1.2)
                    
                #     image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

            # E. 轉 Tensor
            image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
            output["image"] = image_tensor

        # ===========================================================
        # 3. 後續處理 (通用)
        # ===========================================================
        
        output["original_size"] = original_size
        
        # 處理 Ref Void Mask (黑色區域檢測)
        ref_mask_tensor = torch.as_tensor(ref_mask).permute(2, 0, 1).float()
        output["reference_mask"] = ref_mask_tensor
        ref_void_mask = (ref_mask_tensor.sum(dim=0) == 0)
        output["ref_void_mask"] = ref_void_mask
        
        # 處理 GT Mask
        output["gt_mask"] = torch.as_tensor(gt_mask).long()

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