import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os

# 確保路徑正確，指向你修改過的 SAMLoss
from utils.loss import SAMLoss 
from segment_anything.modeling.sam import Sam

class SAMTrainer:
    def __init__(self, model: Sam, train_loader: DataLoader, val_loader: DataLoader, device: str):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 初始化你修改後的 Loss 函數
        self.criterion = SAMLoss(focal_weight=20.0, dice_weight=1.0)
        
        # 使用最新的 torch.amp API
        self.scaler = torch.amp.GradScaler('cuda')
        
        # --- 權重凍結策略 ---
        # 凍結 Image Encoder (ViT 大模型，不建議在一般微調中訓練)
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        
        # 凍結 Prompt Encoder (幾何編碼通常很穩健)
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False
            
        # 僅開啟 Mask Decoder 進行微調
        for param in self.model.mask_decoder.parameters():
            param.requires_grad = True

        # 設定優化器，僅針對需要更新的參數
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
        
        # 學習率排程：當 Loss 不再下降時自動調降
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5
        )

    def train_epoch(self, epoch_index):
        self.model.mask_decoder.train() 
        self.model.prompt_encoder.eval()
        self.model.image_encoder.eval()
        
        # 用於累加整個 epoch 的各項 Loss
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} Training")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_masks = batch['mask'].to(self.device)
            
            # 準備多點提示 (B, N, 2) 與 (B, N)
            point_coords = batch['point_coords'].to(self.device)
            point_labels = batch['point_labels'].to(self.device)
            points = (point_coords, point_labels)

            self.optimizer.zero_grad()

            # 使用混合精度訓練以加速並節省顯存
            with torch.amp.autocast('cuda'):
                # 1. 取得 Image Embedding (凍結狀態)
                with torch.no_grad():
                    image_embeddings = self.model.image_encoder(images)

                # 2. 取得 Prompt Embedding (使用 Point Prompt, 不使用 Boxes)
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=points,
                        boxes=None,
                        masks=None
                    )

                # 3. 取得 Dense Position Embedding
                image_pe = self.model.prompt_encoder.get_dense_pe()
                image_pe = image_pe.repeat(images.shape[0], 1, 1, 1)

                # 4. Decoder 預測
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False, # 單一物件分割
                )

                # 5. 上採樣回原始輸入尺寸 (1024x1024)
                upscaled_masks = F.interpolate(
                    low_res_masks,
                    size=(self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                    mode="bilinear",
                    align_corners=False,
                )

                # 6. 計算 Loss
                loss, loss_dict = self.criterion(upscaled_masks, gt_masks, iou_predictions)
            
            # 反向傳播與優化
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 記錄細項 Loss
            for k, v in loss_dict.items():
                epoch_metrics[k] += v
            
            # 更新進度條顯示
            pbar.set_postfix(
                loss=loss.item(), 
                bce=loss_dict['bce'], 
                dice=loss_dict['dice']
            )

        # 計算平均 Loss
        avg_metrics = {k: v / len(self.train_loader) for k, v in epoch_metrics.items()}
        
        # 更新 Scheduler
        self.scheduler.step(avg_metrics['total'])
        
        return avg_metrics

    def save_checkpoint(self, path):
        # 儲存模型權重 (主要是 mask_decoder 的微調結果)
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")