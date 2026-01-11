import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F  # 補上 F
import os

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
        
        # --- Encord 建議修改重點：優化器設定 ---
        # 1. Image Encoder 絕對凍結 (Forward 時用 no_grad 節省記憶體)
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
            
        # 2. Prompt Encoder 通常也建議凍結，因為幾何位置編碼不需要重訓
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False
            
        # 3. 只訓練 Mask Decoder
        for param in self.model.mask_decoder.parameters():
            param.requires_grad = True

        # 優化器只包含 mask_decoder
        trainable_params = [
            {"params": self.model.mask_decoder.parameters()}
        ]
        
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

    def train_epoch(self, epoch_index):
        # 設定為 train 模式，但因為 requires_grad=False，Encoder 參數不會變
        self.model.mask_decoder.train() 
        self.model.prompt_encoder.eval() # 保持 eval 模式
        self.model.image_encoder.eval()  # 保持 eval 模式
        
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index} Training")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_masks = batch['mask'].to(self.device)
            boxes = batch['box'].to(self.device)

            self.optimizer.zero_grad()

            # --- Forward Pass ---
            
            # 1. Image Encoder: 嚴格使用 no_grad 節省顯存
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(images)

            # 2. Prompt Encoder: 凍結狀態，使用 no_grad
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=boxes.unsqueeze(1), 
                    masks=None
                )

            # 3. Mask Decoder: 這是唯一需要計算梯度的地方
            image_pe = self.model.prompt_encoder.get_dense_pe().to(self.device)
            image_pe = image_pe.repeat(images.shape[0], 1, 1, 1)

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, 
            )

            # 4. Post-process
            upscaled_masks = F.interpolate(
                low_res_masks,
                size=(self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )

            # Loss
            loss, loss_dict = self.criterion(upscaled_masks, gt_masks, iou_predictions)
            
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), bce=loss_dict['bce'], dice=loss_dict['dice'])

        return epoch_loss / len(self.train_loader)

    def save_checkpoint(self, path):
        # 建議只存 Mask Decoder 以節省空間，或者存整個 dict 方便載入
        # 這裡存整個 state_dict 比較通用
        torch.save(self.model.state_dict(), path)