# pair_dataloader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import pandas as pd

class PairSegmentationDataset(Dataset):
    def __init__(self, csv_file: str, image_size: int = 1024, mode: str = 'train',
                 force_raw_images: bool = False):
        """
        Args:
            csv_file (str): CSV 路徑
            image_size (int): 統一縮放尺寸 (預設 1024)
            mode (str): 'train', 'val', 或 'test'
            force_raw_images (bool): 強制使用原始影像，忽略預提取 feature cache。
                當 Image Encoder 解凍訓練時必須設為 True，否則 encoder 被 bypass，梯度無法回傳。
        """
        self.image_size = image_size
        self.mode = mode
        self.force_raw_images = force_raw_images
        
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
            # force_raw_images=True 時不使用快取，不要求 feature_path 非空
            if self.has_cached_features and not self.force_raw_images:
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
        # 1. 判斷是否使用 Cache
        # ===========================================================
        # force_raw_images=True 時強制走原始影像路徑（Image Encoder 解凍訓練必須如此）
        use_cache = False
        if (not self.force_raw_images
                and self.has_cached_features
                and pd.notna(row.get('feature_path'))
                and os.path.exists(str(row.get('feature_path')))):
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

            # [CMAAlignment] 即使使用 cache，也需要原始影像供 VGG16 幾何對齊使用
            if pd.notna(row.get('image_path')) and os.path.exists(str(row.get('image_path', ''))):
                img_raw = cv2.imread(str(row['image_path']))
                if img_raw is not None:
                    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                    img_raw = cv2.resize(img_raw, (self.image_size, self.image_size),
                                         interpolation=cv2.INTER_LANCZOS4)
                    output["image"] = torch.as_tensor(img_raw).permute(2, 0, 1).float()

            ref_col_path = row.get(self.ref_col)
            if pd.notna(ref_col_path) and os.path.exists(str(ref_col_path)):
                ref_raw = cv2.imread(str(ref_col_path))
                if ref_raw is not None:
                    ref_raw = cv2.cvtColor(ref_raw, cv2.COLOR_BGR2RGB)
                    ref_raw = cv2.resize(ref_raw, (self.image_size, self.image_size),
                                          interpolation=cv2.INTER_LANCZOS4)
                    output["clear_image"] = torch.as_tensor(ref_raw).permute(2, 0, 1).float()

            # Ref Mask 讀取 (Cache 模式通常不做增強，直接 resize)
            if pd.notna(row.get(self.ref_col)) and os.path.exists(str(row[self.ref_col])):
                ref_mask = cv2.imread(str(row[self.ref_col]))
                ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
            else:
                # print(f"Reference not found for sample {idx} in cache mode. Using empty mask.")
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
                # print(f"Loading reference from: {row[self.ref_col]}")
                ref_mask = cv2.imread(str(row[self.ref_col]))
                ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
            else:
                print(f"Reference not found for sample {idx}. Using empty mask.")
                ref_mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # C. 讀取 GT Mask
            if pd.notna(row.get('gt_path')) and os.path.exists(str(row['gt_path'])):
                gt_mask = cv2.imread(row['gt_path'], cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = np.full((original_size[0], original_size[1]), 255, dtype=np.uint8)

            # D. 先統一 Resize 到模型輸入尺寸 (例如 1024)
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            # clear_image 與 adverse image 使用相同的 LANCZOS4 resize，確保進入 image_encoder 前品質一致
            clear_image = cv2.resize(ref_mask.copy(), (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            # E. 轉 Tensor
            image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
            output["image"] = image_tensor
            # clear_image：供 CMAAlignment.pre_align() 提取 VGG 特徵使用
            output["clear_image"] = torch.as_tensor(clear_image).permute(2, 0, 1).float()

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

        # 處理 Invalid Mask（optional，目前僅 ACDC 提供）
        # ACDC invalid_mask 實際為二值圖：0 = 有效區域，1 = 無效區域（車頭遮擋等固定盲區）
        if 'invalid_mask' in row and pd.notna(row.get('invalid_mask')) and os.path.exists(str(row['invalid_mask'])):
            inv = cv2.imread(str(row['invalid_mask']), cv2.IMREAD_GRAYSCALE)
            inv = cv2.resize(inv, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            output["invalid_mask"] = torch.as_tensor(inv != 0).bool()  # True = 無效，需排除
        else:
            output["invalid_mask"] = torch.zeros(self.image_size, self.image_size, dtype=torch.bool)

        # 固定使用全部 19 個類別（與推論時保持一致，消除 oracle gap）
        output["text_prompts"] = [self.ID_TO_NAME[i] for i in range(len(self.ID_TO_NAME))]

        # 處理 condition_id（天氣條件索引：fog=0, rain=1, snow=2, night=3）
        # Strict mode：CSV 必須有合法 condition_id，缺失或越界直接 raise
        cid_raw = row.get('condition_id')
        if 'condition_id' not in row or pd.isna(cid_raw):
            raise ValueError(
                f"Sample {idx} 缺少 condition_id 欄位。"
                f"請確認 CSV 有 condition_id 欄位且值 ∈ {{0,1,2,3}}（fog/rain/snow/night）。"
            )
        cid_int = int(cid_raw)
        if cid_int not in (0, 1, 2, 3):
            raise ValueError(
                f"Sample {idx} 的 condition_id={cid_int} 越界，必須為 0(fog)/1(rain)/2(snow)/3(night)。"
            )
        output["condition_id"] = torch.tensor(cid_int, dtype=torch.long)

        # 處理 condition 字串（供 per-condition mIoU 計算使用）
        if 'condition' in row and pd.notna(row.get('condition')):
            output["condition"] = str(row['condition'])
        else:
            output["condition"] = None

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
        condition_ids = torch.stack([item['condition_id'] for item in batch])
        conditions = [item['condition'] for item in batch]  # List[str | None]

        batch_dict = {
            "reference_mask": ref_masks,
            "ref_void_mask": ref_void_masks,
            "gt_mask": gt_masks,
            "invalid_mask": invalid_masks,
            "text_prompts": text_prompts,
            "original_size": original_sizes,
            "condition_id": condition_ids,
            "condition": conditions,
        }

        # 2. 動態處理影像輸入
        if 'image_embedding' in batch[0]:
            batch_dict['image_embedding'] = torch.stack([item['image_embedding'] for item in batch])

        # 原始影像與晴天參考圖：CMAAlignment 的 VGG16 backbone 必須使用，
        # 即使是 cache 模式也需要提供。使用 all() 確保 batch 內全部樣本都有才堆疊。
        if all('image' in item for item in batch):
            batch_dict['image'] = torch.stack([item['image'] for item in batch])
        if all('clear_image' in item for item in batch):
            batch_dict['clear_image'] = torch.stack([item['clear_image'] for item in batch])

        return batch_dict
