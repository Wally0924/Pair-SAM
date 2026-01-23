# weather_trainer.py
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
        
        # 初始化 Loss
        self.criterion = SAMLoss(focal_weight=5.0, dice_weight=2.0, iou_weight=1.0)
        
        # 使用混合精度訓練 (AMP)
        self.scaler = torch.amp.GradScaler('cuda')
        
        # --- 1. 權重凍結策略 ---
        # 確保 Image/Text Encoder 凍結
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False
            
        # --- 2. 解鎖需要訓練的模組 ---
        # 包含 Mask Encoder, Fusion, Gate, Decoder, Prompt Encoder (Adapter)
        trainable_modules = [
            self.model.mask_encoder,
            self.model.fusion_module,
            self.model.gate_module,
            self.model.mask_decoder,
            self.model.prompt_encoder, 
        ]
        
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
                
        # 位置編碼
        self.model.pe_layer.requires_grad = True

        # 收集優化參數
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"✅ 模型初始化完成，可訓練參數數量: {len(trainable_params)}")
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        
        # 學習率排程
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def _prepare_batch_input(self, batch, batch_size):
        """
        輔助函式：將 DataLoader 的 Batch 轉換為模型需要的 List[Dict] 格式
        自動判斷是 Raw Image 還是 Cached Embedding
        """
        batched_input = []
        
        # 檢查是否使用快取特徵
        use_cached_features = 'image_embedding' in batch
        
        for i in range(batch_size):
            input_dict = {
                'reference_mask': batch['reference_mask'][i].to(self.device),
                'text_prompts': batch['text_prompts'][i], # List[str]
                'original_size': batch['original_size'][i]
            }
            
            # 關鍵修改：動態選擇輸入源
            if use_cached_features:
                input_dict['image_embedding'] = batch['image_embedding'][i].to(self.device)
            else:
                input_dict['image'] = batch['image'][i].to(self.device)
            
            batched_input.append(input_dict)
            
        return batched_input

    def train_epoch(self, epoch_index):
        self.model.train()
        self.model.image_encoder.eval() # 始終保持 eval (因為凍結)
        self.model.text_encoder.eval()
        
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            
            # 1. 準備輸入資料 (使用新的輔助函式)
            batched_input = self._prepare_batch_input(batch, batch_size)
            
            # GT Masks (修正 key 為 'gt_mask')
            gt_masks = batch['gt_mask'].to(self.device)

            self.optimizer.zero_grad()

            # 2. 混合精度前向傳播
            with torch.amp.autocast('cuda'):
                # Forward Pass
                # 模型內部會根據 key 是 image 還是 image_embedding 自動處理
                outputs = self.model(batched_input, multimask_output=True)
                
                # 3. 計算 Loss
                total_loss = 0
                loss_dict_accum = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
                
                for i in range(batch_size):
                    # 取出 Logits 並上採樣
                    low_res_logits = outputs[i]['low_res_logits']
                    full_res_logits = F.interpolate(
                        low_res_logits,
                        size=(1024, 1024),
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    # 計算 Loss
                    sample_loss, sample_dict = self.criterion(
                        pred_masks=full_res_logits,
                        gt_mask=gt_masks[i],
                        iou_predictions=outputs[i]['iou_predictions'],
                        text_prompts=batched_input[i]['text_prompts']
                    )
                    
                    total_loss += sample_loss
                    for k, v in sample_dict.items():
                        loss_dict_accum[k] += v

                # 平均 Batch Loss
                total_loss = total_loss / batch_size

            # 4. 反向傳播
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 5. 更新統計
            step_count += 1
            for k in epoch_metrics:
                epoch_metrics[k] += (loss_dict_accum[k] / batch_size)
            
            pbar.set_postfix(
                loss=total_loss.item(), 
                dice=(loss_dict_accum['dice']/batch_size)
            )

        avg_metrics = {k: v / step_count for k, v in epoch_metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def validate_epoch(self, epoch_index):
        self.model.eval()
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_index+1} [Val]")
        step_count = 0
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            
            # 1. 準備輸入
            batched_input = self._prepare_batch_input(batch, batch_size)
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
        
        # 根據 Validation Loss 更新 LR
        self.scheduler.step(avg_metrics['total'])
        
        return avg_metrics

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Checkpoint saved to {path}")