import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
from PIL import Image

class WeatherSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 1024, mode: str = 'train'):
        """
        Args:
            root_dir (str): 資料集根目錄，應包含 bad_weather_images/, reference_masks/, labels/
            image_size (int): 統一縮放尺寸 (SAM 預設 1024)
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        
        # 定義資料夾路徑
        self.images_path = os.path.join(root_dir, "bad_weather_images")
        self.ref_masks_path = os.path.join(root_dir, "reference_masks")
        self.labels_path = os.path.join(root_dir, "labels")
        
        # 讀取檔案列表 (假設檔名是對應的)
        self.image_files = sorted([
            f for f in os.listdir(self.images_path) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # 定義類別 ID 對應 (依據您的資料集 Cityscapes 或其他)
        # 格式: "類別名稱": ID
        self.CLASS_MAP = {
            "road": 0,
            "sidewalk": 1,
            "building": 2,
            "wall": 3,
            "fence": 4,
            "pole": 5,
            "traffic light": 6,
            "traffic sign": 7,
            "vegetation": 8,
            "terrain": 9,
            "sky": 10,
            "person": 11,
            "rider": 12,
            "car": 13,
            "truck": 14,
            "bus": 15,
            "train": 16,
            "motorcycle": 17,
            "bicycle": 18
        }
        # 反向映射 ID -> Name
        self.ID_TO_NAME = {v: k for k, v in self.CLASS_MAP.items()}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 1. 讀取惡劣天氣影像 (Bad Image)
        img_path = os.path.join(self.images_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 讀取參考遮罩 (Reference Mask)
        # 假設檔名與影像相同，或者是 png 格式
        mask_name = os.path.splitext(img_name)[0] + ".png"
        ref_mask_path = os.path.join(self.ref_masks_path, mask_name)
        
        # 如果找不到參考遮罩，給一個全黑的 (避免程式崩潰)
        if os.path.exists(ref_mask_path):
            ref_mask = cv2.imread(ref_mask_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        else:
            ref_mask = np.zeros_like(image)

        # 3. 讀取 Ground Truth Label (類別索引圖)
        label_path = os.path.join(self.labels_path, mask_name)
        gt_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 讀取為灰階 ID

        # --- Resize ---
        original_size = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # --- Transform to Tensor ---
        # 轉為 (3, H, W) 且範圍 0~1，但不做 Mean/Std Normalize
        # 因為 WeatherSAM.preprocess 會做 Normalize
        image_tensor = torch.as_tensor(image).permute(2, 0, 1).float() 
        ref_mask_tensor = torch.as_tensor(ref_mask).permute(2, 0, 1).float()
        gt_tensor = torch.as_tensor(gt_mask).long() # (H, W)

        # --- 生成 Text Prompts ---
        # 策略：找出這張圖 GT 中有哪些類別，就送這些文字進去訓練
        unique_classes = np.unique(gt_mask)
        active_prompts = []
        
        for cls_id in unique_classes:
            if cls_id in self.ID_TO_NAME:
                active_prompts.append(self.ID_TO_NAME[cls_id])
        
        # 如果圖中沒有已知類別 (例如全是 ignore index)，隨機給一個避免空 list
        if not active_prompts:
            active_prompts = ["road"] 

        # 為了訓練穩定，每次可以隨機採樣 K 個 prompt (例如最多 3 個)，避免 OOM
        if self.mode == 'train' and len(active_prompts) > 3:
            active_prompts = random.sample(active_prompts, 3)

        return {
            "image": image_tensor,          # (3, 1024, 1024)
            "reference_mask": ref_mask_tensor, # (3, 1024, 1024)
            "gt_mask": gt_tensor,           # (1024, 1024)
            "text_prompts": active_prompts, # List[str] e.g. ["car", "road"]
            "original_size": original_size
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        自定義 Collate Function，因為 text_prompts 是 list of list，
        預設的 collate 可能會報錯或行為不如預期。
        """
        images = torch.stack([item['image'] for item in batch])
        ref_masks = torch.stack([item['reference_mask'] for item in batch])
        gt_masks = torch.stack([item['gt_mask'] for item in batch])
        
        # Text prompts 和 original_size 保持列表形式
        text_prompts = [item['text_prompts'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        return {
            "image": images,
            "reference_mask": ref_masks,
            "gt_mask": gt_masks,
            "text_prompts": text_prompts,
            "original_size": original_sizes
        }