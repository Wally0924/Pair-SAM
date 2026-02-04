# weather_trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# 確保引用路徑正確
from utils.new_loss import SAMLoss
from segment_anything.modeling import WeatherSAM 

class WeatherSAMTrainer:
    def __init__(
        self, 
        model: WeatherSAM, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        args=None
    ):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = args.device
        self.args = args
        
        # [修改] 從 args 讀取 Loss 權重，若無則使用預設值
        f_w = args.focal_weight if args else 2.0
        d_w = args.dice_weight if args else 2.0
        i_w = args.iou_weight if args else 1.0
        ls = args.label_smoothing if args else 0.1
        lr = args.lr if args else 1e-4
        
        print(f"📉 Initializing Loss with: Focal={f_w}, Dice={d_w}, IoU={i_w}, Smooth={ls}")
        self.criterion = SAMLoss(focal_weight=f_w, dice_weight=d_w, iou_weight=i_w, label_smoothing=ls)
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 凍結與解凍策略
        for param in self.model.parameters():
            param.requires_grad = False
            
        trainable_modules = [
            self.model.mask_encoder,
            self.model.fusion_module,
            self.model.gate_module,
            self.model.mask_decoder,
            self.model.prompt_encoder,
            self.model.location_encoder,
        ]
        
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        self.model.pe_layer.requires_grad = True

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"✅ 總可訓練參數數量: {len(trainable_params)}")
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

        # ==========================================
        # [修改] 實作 Warmup + Cosine Decay 策略
        # ==========================================
        num_epochs = args.epochs if args else 50
        warmup_epochs = 5

        def lr_lambda(epoch_idx):
            # 1. Warmup 階段: 線性上升
            if epoch_idx < warmup_epochs:
                # 例如第 0 epoch 返回 0，第 5 epoch 返回 1.0
                return float(epoch_idx + 1) / float(warmup_epochs)
            
            # 2. Cosine Decay 階段: 緩慢下降
            else:
                # 計算進度 (0.0 ~ 1.0)
                progress = float(epoch_idx - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
                # 餘弦公式: 0.5 * (1 + cos(pi * progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        os.makedirs("debug_viz", exist_ok=True)

    def _prepare_batch_input(self, batch, batch_size):
        batched_input = []
        use_cached_features = 'image_embedding' in batch
        
        for i in range(batch_size):
            input_dict = {
                'reference_mask': batch['reference_mask'][i].to(self.device),
                'ref_void_mask': batch['ref_void_mask'][i].to(self.device),
                'text_prompts': batch['text_prompts'][i], 
                'original_size': batch['original_size'][i],
                'location': batch['location'][i].to(self.device) 
            }
            if use_cached_features:
                input_dict['image_embedding'] = batch['image_embedding'][i].to(self.device)
            else:
                input_dict['image'] = batch['image'][i].to(self.device)
            batched_input.append(input_dict)
        return batched_input

    def train_epoch(self, epoch_index):
        self.model.train()
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
        
        # 取得 max_norm 設定
        max_norm = self.args.max_norm if self.args else 0.3
        
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
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            
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
                self._save_debug_snapshot(first_batch_logits, epoch_index, step_count)

        self.scheduler.step()
        avg_metrics = {k: v / step_count for k, v in epoch_metrics.items()}
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"   🔄 Learning Rate Updated: {current_lr:.2e}")
        
        return avg_metrics

    def _save_debug_snapshot(self, logits, epoch, step):
        pred_logit = logits[0, 0, :, :]
        mask_viz = torch.sigmoid(pred_logit).detach().cpu().numpy()
        save_path = f"debug_viz/epoch_{epoch+1}_step_{step}.png"
        plt.imsave(save_path, mask_viz, cmap='gray')
        max_val = mask_viz.max()
        status = "🟢 OK" if max_val > 0.1 else "🔴 Collapsed"
        # print(f"   📸 Snapshot saved! Max Value: {max_val:.4f} [{status}]")

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
                
                loss_dict_accum = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
                
                for i in range(batch_size):
                    low_res_logits = outputs[i]['low_res_logits']
                    full_res_logits = F.interpolate(
                        low_res_logits, size=(1024, 1024), mode="bilinear", align_corners=False
                    )
                    
                    _, sample_dict = self.criterion(
                        pred_masks=full_res_logits,
                        gt_mask=gt_masks[i],
                        iou_predictions=outputs[i]['iou_predictions'],
                        text_prompts=batched_input[i]['text_prompts']
                    )
                    
                    for k, v in sample_dict.items():
                        loss_dict_accum[k] += v
            
            step_count += 1
            for k in epoch_metrics:
                epoch_metrics[k] += (loss_dict_accum[k] / batch_size)
                
        avg_metrics = {k: v / step_count for k, v in epoch_metrics.items()}
        # self.scheduler.step(avg_metrics['total'])
        return avg_metrics

    def save_checkpoint(self, path, epoch=None, best_score=None):
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_score': best_score
        }
        # 儲存參數配置以便未來查看
        if self.args:
            if hasattr(self.args, '__dict__'):
                checkpoint_dict['config'] = vars(self.args)
            else:
                checkpoint_dict['config'] = self.args
        else:
            checkpoint_dict['config'] = {}
            
        torch.save(checkpoint_dict, path)
        print(f"💾 Checkpoint saved to {path}")