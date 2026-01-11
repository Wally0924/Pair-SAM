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
        self.image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith(('.jpg', '.png'))])
        
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        # 假設 mask 為 png，且檔名對應
        mask_path = os.path.join(self.masks_path, img_name.replace(".jpg", ".png").replace(".jpeg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        # Resize (實際專案建議保持長寬比 padding，這裡簡化為 resize)
        original_size = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # 轉 Tensor 並標準化
        image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
        image_tensor = (image_tensor - self.pixel_mean) / self.pixel_std
        
        mask_tensor = torch.as_tensor(mask).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        # --- Encord 建議修改重點：生成更魯棒的 Prompt ---
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 加大擾動範圍：Encord 建議讓 Box 稍微不準一點，訓練 Decoder 修正能力
            # 這裡設定 0~20 pixels 的隨機擴張/收縮
            perturbation = 20 
            x_min = max(0, x_min - np.random.randint(0, perturbation))
            x_max = min(self.image_size, x_max + np.random.randint(0, perturbation))
            y_min = max(0, y_min - np.random.randint(0, perturbation))
            y_max = min(self.image_size, y_max + np.random.randint(0, perturbation))
            
            box = np.array([x_min, y_min, x_max, y_max])
        else:
            # 處理無物件的情況
            box = np.array([0, 0, self.image_size, self.image_size])

        box_tensor = torch.as_tensor(box).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor.unsqueeze(0),
            "box": box_tensor,
            "original_size": original_size
        }