# weather_trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

# 確保引用路徑正確
from utils.new_loss import SAMLoss
from segment_anything.modeling import WeatherSAM 

class WeatherSAMTrainer:
    def __init__(
        self, 
        model: WeatherSAM, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        device: str,
        lr: float = 5e-5,
        args = None  # [新增 1] 接收參數設定
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args # [新增 2] 儲存參數設定
        
        # Loss 權重策略
        self.criterion = SAMLoss(focal_weight=2.0, dice_weight=2.0, iou_weight=1.0)
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 解凍策略 (Unfreezing Strategy)
        for param in self.model.parameters():
            param.requires_grad = False
            
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

        # 統計可訓練參數
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"✅ 總可訓練參數數量: {len(trainable_params)}")
        
        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        os.makedirs("debug_viz", exist_ok=True)

    def _prepare_batch_input(self, batch, batch_size):
        # ... (保持不變) ...
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
        # ... (保持不變) ...
        self.model.train()
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            batched_input = self._prepare_batch_input(batch, batch_size)
            gt_masks = batch['gt_mask'].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = self.model(batched_input, multimask_output=True)
                
                total_loss = 0
                loss_dict_accum = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
                first_batch_logits = None 

                for i in range(batch_size):
                    low_res_logits = outputs[i]['low_res_logits']
                    full_res_logits = F.interpolate(
                        low_res_logits,
                        size=(1024, 1024),
                        mode="bilinear",
                        align_corners=False
                    )

                    if i == 0:
                        first_batch_logits = full_res_logits
                    
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

            if step_count % 1000 == 0 and first_batch_logits is not None:
                pred_logit = first_batch_logits[0, 0, :, :]
                mask_viz = torch.sigmoid(pred_logit).detach().cpu().numpy()
                save_path = f"debug_viz/epoch_{epoch_index+1}_step_{step_count}.png"
                plt.imsave(save_path, mask_viz, cmap='gray')
                max_val = mask_viz.max()
                status = "🟢 OK" if max_val > 0.1 else "🔴 Collapsed (Black)"
                print(f"   📸 Snapshot saved! Max Value: {max_val:.4f} [{status}]")

        avg_metrics = {k: v / step_count for k, v in epoch_metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def validate_epoch(self, epoch_index):
        # ... (保持不變) ...
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

    def save_checkpoint(self, path, epoch=None, best_score=None):
        """
        [修改 3] 儲存 Checkpoint 與完整 Config
        """
        # 1. 準備要儲存的字典
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_score': best_score
        }
        
        # 2. 如果有 Config (args)，轉成 dict 存入
        if self.args:
            # 判斷是否為 argparse.Namespace，若是則轉 dict
            if hasattr(self.args, '__dict__'):
                checkpoint_dict['config'] = vars(self.args)
            else:
                checkpoint_dict['config'] = self.args
        else:
            checkpoint_dict['config'] = {}
            
        # 3. 儲存
        torch.save(checkpoint_dict, path)
        print(f"💾 Checkpoint (with Config) saved to {path}")