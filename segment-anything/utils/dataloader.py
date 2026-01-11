import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
from torchvision.transforms import functional as F

class WeatherSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 1024, mode: str = 'train', max_images: int = None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        
        self.images_path = os.path.join(root_dir, "images")
        self.masks_path = os.path.join(root_dir, "masks")
        
        all_image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if max_images is not None and len(all_image_files) > max_images:
            random.seed(42)
            random.shuffle(all_image_files)
            self.image_files = all_image_files[:max_images]
            print(f"[{mode}] 測試模式：已從 {len(all_image_files)} 張圖中隨機選取 {len(self.image_files)} 張進行訓練。")
        else:
            self.image_files = all_image_files
            print(f"[{mode}] 全量模式：載入所有 {len(self.image_files)} 張圖片。")
        
        self.TARGET_CLASSES = {
            "road":     [128, 64, 128],   
            "sidewalk": [244, 35, 232],   
            "lane":     [157, 234, 50]    
        }
        
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_path, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        original_size = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        available_classes = []
        for class_name, rgb in self.TARGET_CLASSES.items():
            class_mask = np.all(mask == rgb, axis=2)
            threshold = 50 if class_name == 'lane' else 400
            if np.sum(class_mask) > threshold:
                available_classes.append(class_name)
        
        binary_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        is_small_object = False # 標記是否為小物體 (車道線)
        
        if len(available_classes) > 0:
            target_class = random.choice(available_classes)
            target_rgb = self.TARGET_CLASSES[target_class]
            raw_mask = np.all(mask == target_rgb, axis=2).astype(np.uint8)
            
            if target_class == 'lane':
                is_small_object = True
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
                valid_components = [i for i in range(1, num_labels) if stats[i, 4] > 50]
                if len(valid_components) > 0:
                    chosen_label = random.choice(valid_components)
                    binary_mask = (labels == chosen_label).astype(np.float32)
            else:
                binary_mask = raw_mask.astype(np.float32)

        image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()
        image_tensor = (image_tensor - self.pixel_mean) / self.pixel_std
        mask_tensor = torch.as_tensor(binary_mask).float().unsqueeze(0)

        # ==========================================
        # 5. 生成多點提示 (Multi-point Prompt)
        # ==========================================
        y_indices, x_indices = np.where(binary_mask > 0)
        
        # 設定要採樣幾個點
        # 如果是小物體(車道線) -> 1個點
        # 如果是大物體(路面) -> 3~5個點
        num_points = 1 if is_small_object else random.randint(3, 5)
        
        # 預留固定的 Tensor 大小 (例如最多 5 個點)
        # SAM 批次訓練時，Tensor 大小必須一致，不足的補 (-1, -1)
        MAX_POINTS = 5
        
        coords = np.zeros((MAX_POINTS, 2), dtype=np.float32)
        labels = np.ones(MAX_POINTS, dtype=np.float32) * -1 # 初始化為 -1 (Padding)
        
        if len(y_indices) > 0:
            # 隨機取樣 num_points 個索引 (可重複取樣，或者不重複)
            choose_indices = np.random.choice(len(y_indices), size=num_points, replace=True)
            
            for i, rand_idx in enumerate(choose_indices):
                coords[i] = [x_indices[rand_idx], y_indices[rand_idx]]
                labels[i] = 1 # 前景點
        else:
            # 空 Mask，全部維持 Padding 狀態，或者給一個負樣本
            coords[0] = [0, 0]
            labels[0] = -1 

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "point_coords": torch.as_tensor(coords).float(), # Shape: (5, 2)
            "point_labels": torch.as_tensor(labels).float(), # Shape: (5,)
            "original_size": original_size
        }