import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision.transforms import functional as F

class WeatherSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 1024, mode: str = 'train'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        
        self.images_path = os.path.join(root_dir, "images")
        self.masks_path = os.path.join(root_dir, "masks")
        # 支援常見影像格式
        self.image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # SAM 的標準化參數
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        
        # 尋找對應的 mask 檔名 (假設 mask 是 png)
        # 根據你的檔名範例，影像如果是 02878323.jpg，mask 是 02878323.png
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_path, mask_name)

        # 1. 讀取影像 (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 讀取 Mask (RGB 彩色圖，不可轉灰階)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # 轉 RGB 以確保顏色對應正確

        # 3. Resize
        # ⚠️ 重點：RGB Mask Resize 必須用 INTER_NEAREST (最近鄰插值)
        # 如果用線性插值，紅色跟藍色邊界會混出紫色，產生不存在的類別！
        original_size = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # 4. 處理 RGB Mask：隨機挑選一個顏色作為目標
        # 找出圖中所有出現的唯一顏色 (排除背景黑色 [0,0,0])
        # mask shape: (H, W, 3) -> reshape -> (N, 3)
        reshaped_mask = mask.reshape(-1, 3)
        unique_colors = np.unique(reshaped_mask, axis=0)
        
        # 過濾掉黑色背景 (假設 [0,0,0] 是背景不需訓練)
        valid_colors = [c for c in unique_colors if not np.array_equal(c, [0, 0, 0])]
        
        if len(valid_colors) > 0:
            # 隨機選一個顏色 (例如這次選「粉紅色路面」)
            target_color = valid_colors[np.random.randint(len(valid_colors))]
            
            # 製作二值化 Mask: 只有該顏色的地方是 1，其他是 0
            # np.all(axis=2) 比較 R,G,B 三個通道是否都相等
            binary_mask = np.all(mask == target_color, axis=2).astype(np.float32)
        else:
            # 如果整張圖都是黑的 (無標註)
            binary_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # 5. 轉換 Image Tensor
        image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
        image_tensor = (image_tensor - self.pixel_mean) / self.pixel_std
        
        mask_tensor = torch.as_tensor(binary_mask).float()

        # 6. 生成 Prompt (Box)
        y_indices, x_indices = np.where(binary_mask > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 隨機擾動 (Perturbation)
            perturbation = 20 
            x_min = max(0, x_min - np.random.randint(0, perturbation))
            x_max = min(self.image_size, x_max + np.random.randint(0, perturbation))
            y_min = max(0, y_min - np.random.randint(0, perturbation))
            y_max = min(self.image_size, y_max + np.random.randint(0, perturbation))
            
            box = np.array([x_min, y_min, x_max, y_max])
        else:
            # Fallback
            box = np.array([0, 0, self.image_size, self.image_size])

        box_tensor = torch.as_tensor(box).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor.unsqueeze(0), # (1, H, W)
            "box": box_tensor,
            "original_size": original_size
        }