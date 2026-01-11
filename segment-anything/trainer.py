import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
from torch.cuda.amp import autocast, GradScaler 

# 假設 loss.py 已經放在 utils 資料夾下
from utils.loss import SAMLoss 
from segment_anything.modeling.sam import Sam

class SAMTrainer:
    def __init__(self, model: Sam, train_loader: DataLoader, val_loader: DataLoader, device: str):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = SAMLoss()
        
        # --- 初始化 GradScaler (用於混合精度訓練) ---
        self.scaler = torch.amp.GradScaler('cuda')
        
        # --- 凍結 Encoder ---
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False
            
        # 只訓練 Mask Decoder
        for param in self.model.mask_decoder.parameters():
            param.requires_grad = True

        trainable_params = [
            {"params": self.model.mask_decoder.parameters()}
        ]
        
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

    def train_epoch(self, epoch_index):
        self.model.mask_decoder.train() 
        self.model.prompt_encoder.eval()
        self.model.image_encoder.eval()
        
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} Training")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_masks = batch['mask'].to(self.device)
            
            # === 修改重點：改用 Point Prompt ===
            # 取得點座標與標籤
            point_coords = batch['point_coords'].to(self.device) # (B, N, 2)
            point_labels = batch['point_labels'].to(self.device) # (B, N)
            
            # SAM 要求的 points 格式是一個 tuple: (coords, labels)
            points = (point_coords, point_labels)

            self.optimizer.zero_grad()

            # --- Forward Pass (使用 AMP) ---
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    image_embeddings = self.model.image_encoder(images)

                with torch.no_grad():
                    # 這裡傳入 points，將 boxes 設為 None
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=points, # 傳入點提示
                        boxes=None,    # 不使用框
                        masks=None
                    )

                image_pe = self.model.prompt_encoder.get_dense_pe().to(self.device)
                image_pe = image_pe.repeat(images.shape[0], 1, 1, 1)

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False, 
                )

                upscaled_masks = F.interpolate(
                    low_res_masks,
                    size=(self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                    mode="bilinear",
                    align_corners=False,
                )

                loss, loss_dict = self.criterion(upscaled_masks, gt_masks, iou_predictions)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), bce=loss_dict['bce'], dice=loss_dict['dice'])

        return epoch_loss / len(self.train_loader)

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)