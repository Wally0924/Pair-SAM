# weather_trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np

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
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # [修改] Focal Weight 從 20.0 降至 2.0，避免過度激進導致 Logit 飽和
        self.criterion = SAMLoss(focal_weight=0.5, dice_weight=5.0, iou_weight=1.0)
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Freeze Backbones
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False
            
        # Unlock Trainable Modules
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
        self.model.pe_layer.requires_grad = True

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"✅ 可訓練參數數量: {len(trainable_params)}")
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def _prepare_batch_input(self, batch, batch_size):
        batched_input = []
        use_cached_features = 'image_embedding' in batch
        
        for i in range(batch_size):
            input_dict = {
                'reference_mask': batch['reference_mask'][i].to(self.device),
                'text_prompts': batch['text_prompts'][i], 
                'original_size': batch['original_size'][i]
            }
            if use_cached_features:
                input_dict['image_embedding'] = batch['image_embedding'][i].to(self.device)
            else:
                input_dict['image'] = batch['image'][i].to(self.device)
            batched_input.append(input_dict)
        return batched_input

    def train_epoch(self, epoch_index):
        self.model.train()
        self.model.image_encoder.eval() 
        self.model.text_encoder.eval()
        
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            batched_input = self._prepare_batch_input(batch, batch_size)
            gt_masks = batch['gt_mask'].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # outputs 是一個 List，長度為 Batch Size
                outputs = self.model(batched_input, multimask_output=True)
                
                total_loss = 0
                loss_dict_accum = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
                
                for i in range(batch_size):
                    # 取出 Logits: (K, 3, 256, 256)
                    low_res_logits = outputs[i]['low_res_logits']
                    
                    # 上採樣至 GT 尺寸 (1024x1024)
                    full_res_logits = F.interpolate(
                        low_res_logits,
                        size=(1024, 1024),
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    # 計算 Loss (針對該影像的所有 Prompts)
                    sample_loss, sample_dict = self.criterion(
                        pred_masks=full_res_logits,
                        gt_mask=gt_masks[i],
                        iou_predictions=outputs[i]['iou_predictions'],
                        text_prompts=batched_input[i]['text_prompts']
                    )
                    
                    total_loss += sample_loss
                    for k, v in sample_dict.items():
                        loss_dict_accum[k] += v

                total_loss = total_loss / batch_size

            self.scaler.scale(total_loss).backward()
            
            # [新增] 梯度裁減: 防止 Loss 爆炸導致全白輸出
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step_count += 1
            for k in epoch_metrics:
                val_to_add = loss_dict_accum[k]
                if torch.is_tensor(val_to_add):
                    val_to_add = val_to_add.item()
                epoch_metrics[k] += (val_to_add / batch_size)

            pbar.set_postfix(
                loss=total_loss.item(), 
                dice=(loss_dict_accum['dice']/batch_size)
            )
        if step_count % 100 == 0:
            print(f"   [Step {step_count}] Mean IoU Conf: {outputs[0]['iou_predictions'].mean().item():.4f}")
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
        self.scheduler.step(avg_metrics['total'])
        return avg_metrics

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Checkpoint saved to {path}")