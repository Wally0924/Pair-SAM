import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np

# 引入自定義模組
from utils.new_loss import SAMLoss
from segment_anything.modeling import WeatherSAM 

class WeatherSAMTrainer:
    def __init__(
        self, 
        model: WeatherSAM, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        device: str,
        lr: float = 1e-4
    ):
        """
        初始化 WeatherSAM 訓練器
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 初始化 Loss (確保參數與 loss.py 一致)
        self.criterion = SAMLoss(focal_weight=20.0, dice_weight=1.0, iou_weight=1.0)
        
        # 使用混合精度訓練 (AMP)
        self.scaler = torch.amp.GradScaler('cuda')
        
        # --- 1. 權重凍結策略 ---
        # 凍結 Image Encoder (ViT) 與 Text Encoder (CLIP)
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False
            
        # --- 2. 解鎖需要訓練的模組 ---
        trainable_modules = [
            self.model.mask_encoder,   # Reference Mask 編碼器
            self.model.fusion_module,  # Cross-Attention
            self.model.gate_module,    # Gated Fusion
            self.model.mask_decoder,   # Mask Decoder
            self.model.prompt_encoder, # 如果有 Mask downscaling 層
        ]
        
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
                
        # 特別確保位置編碼也是可訓練的
        self.model.pe_layer.requires_grad = True

        # 收集優化參數
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"✅ 模型初始化完成，可訓練參數數量: {len(trainable_params)}")
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        
        # 學習率排程: Loss 不降時自動減少 LR
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def train_epoch(self, epoch_index):
        """
        執行一個 Training Epoch
        """
        self.model.train()
        # 保持凍結模組在 eval 模式 (例如 BatchNorm)
        self.model.image_encoder.eval()
        self.model.text_encoder.eval()
        
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
        
        for batch in pbar:
            # 1. 資料準備：轉換為 List[Dict] 格式以符合 WeatherSAM 輸入
            batched_input = []
            batch_size = len(batch['text_prompts']) # 實際 Batch Size
            
            for i in range(batch_size):
                batched_input.append({
                    'image': batch['image'][i].to(self.device),
                    'reference_mask': batch['reference_mask'][i].to(self.device),
                    'text_prompts': batch['text_prompts'][i], # List[str]
                    'original_size': batch['original_size'][i]
                })
            
            # GT Masks: (B, 1024, 1024) - 整數類別 ID
            gt_masks = batch['gt_mask'].to(self.device)

            self.optimizer.zero_grad()

            # 2. 混合精度前向傳播
            with torch.amp.autocast('cuda'):
                # Forward Pass
                outputs = self.model(batched_input, multimask_output=True)
                
                # 3. 計算 Loss
                # 由於 WeatherSAM 針對每張圖可能有不同數量的 prompt，
                # 我們採用逐張計算 Loss 再平均的策略。
                total_loss = 0
                loss_dict_accum = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
                
                for i in range(batch_size):
                    # 取出 Logits (K, 3, 256, 256) 並上採樣回 (1024, 1024)
                    # Loss Function 需要 Float Logits 來計算 BCE
                    low_res_logits = outputs[i]['low_res_logits']
                    full_res_logits = F.interpolate(
                        low_res_logits,
                        size=(1024, 1024),
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    # 呼叫 SAMLoss
                    # 注意：參數名稱需對應 loss.py 的定義
                    sample_loss, sample_dict = self.criterion(
                        pred_masks=full_res_logits,                 # (K, 3, 1024, 1024)
                        gt_mask=gt_masks[i],                        # (1024, 1024) [ID Map]
                        iou_predictions=outputs[i]['iou_predictions'], # (K, 3)
                        text_prompts=batched_input[i]['text_prompts']  # List[str]
                    )
                    
                    total_loss += sample_loss
                    for k, v in sample_dict.items():
                        loss_dict_accum[k] += v

                # 平均 Batch Loss
                total_loss = total_loss / batch_size

            # 4. 反向傳播與優化
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 5. 更新統計數據
            step_count += 1
            for k in epoch_metrics:
                # 這裡除以 batch_size 是為了還原單張平均，再累加
                epoch_metrics[k] += (loss_dict_accum[k] / batch_size)
            
            # 更新進度條
            pbar.set_postfix(
                loss=total_loss.item(), 
                dice=(loss_dict_accum['dice']/batch_size)
            )

        # 計算整個 Epoch 的平均指標
        avg_metrics = {k: v / step_count for k, v in epoch_metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def validate_epoch(self, epoch_index):
        """
        執行一個 Validation Epoch (不更新權重)
        """
        self.model.eval()
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_index+1} [Val]")
        step_count = 0
        
        for batch in pbar:
            # 資料準備 (同 Train)
            batched_input = []
            batch_size = len(batch['text_prompts'])
            for i in range(batch_size):
                batched_input.append({
                    'image': batch['image'][i].to(self.device),
                    'reference_mask': batch['reference_mask'][i].to(self.device),
                    'text_prompts': batch['text_prompts'][i],
                    'original_size': batch['original_size'][i]
                })
            gt_masks = batch['gt_mask'].to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model(batched_input, multimask_output=True)
                
                total_loss = 0
                loss_dict_accum = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
                
                for i in range(batch_size):
                    low_res_logits = outputs[i]['low_res_logits']
                    full_res_logits = F.interpolate(
                        low_res_logits, size=(1024, 1024), mode="bilinear", align_corners=False
                    )
                    
                    sample_loss, sample_dict = self.criterion(
                        pred_masks=full_res_logits,
                        gt_mask=gt_masks[i],
                        iou_predictions=outputs[i]['iou_predictions'],
                        text_prompts=batched_input[i]['text_prompts']
                    )
                    
                    total_loss += sample_loss
                    for k, v in sample_dict.items():
                        loss_dict_accum[k] += v
            
            step_count += 1
            for k in epoch_metrics:
                epoch_metrics[k] += (loss_dict_accum[k] / batch_size)
                
        avg_metrics = {k: v / step_count for k, v in epoch_metrics.items()}
        
        # 驗證結束後，更新 Scheduler (根據 Val Loss)
        self.scheduler.step(avg_metrics['total'])
        
        return avg_metrics

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Checkpoint saved to {path}")