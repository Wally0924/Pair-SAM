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
        lr: float = 1e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # [修改 1] Loss 權重策略：降低 Focal (背景懲罰)，大幅提高 Dice (形狀獎勵)
        # 這是為了解決 "Background Collapse" (模型預測全黑) 的問題
        self.criterion = SAMLoss(focal_weight=2.0, dice_weight=2.0, iou_weight=1.0)
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # -------------------------------------------------------
        # [修改 2] 解凍策略 (Unfreezing Strategy)
        # -------------------------------------------------------
        
        # 1. 先凍結所有參數
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. 解凍我們自己加的 Adapter Modules (MaskEncoder, Fusion, etc.)
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
        
        # Positional Encoding 也要訓練
        self.model.pe_layer.requires_grad = True

        # 3. [關鍵] 解凍 ViT Image Encoder 的最後 2 個 Block
        # 這是為了讓模型學會處理 "霧/雨" 造成的 Domain Shift
        # if hasattr(self.model.image_encoder, 'blocks'):
        #     # 取出最後兩層 (無論是 ViT-B 還是 ViT-H 都通用)
        #     layers_to_unfreeze = [
        #         self.model.image_encoder.blocks[-1],
        #         self.model.image_encoder.blocks[-2],
        #         self.model.image_encoder.blocks[-3], # 新增
        #         self.model.image_encoder.blocks[-4]  # 新增
        #     ]
        #     print("🔓 Unfreezing the last 4 blocks of Image Encoder for domain adaptation.")
        #     for layer in layers_to_unfreeze:
        #         for param in layer.parameters():
        #             param.requires_grad = True
        
        # -------------------------------------------------------
        
        # 統計可訓練參數
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"✅ 總可訓練參數數量: {len(trainable_params)}")
        
        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # 建立 Debug 圖片存檔目錄
        os.makedirs("debug_viz", exist_ok=True)

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
        # 注意: 雖然 model.train()，但前面設為 requires_grad=False 的層數不會更新
        # 對於 ViT 內的 BatchNorm/Dropout，若不想啟用隨機性，可額外呼叫 self.model.image_encoder.eval()
        # 但通常微調時保持 train mode 是可以的。
        
        epoch_metrics = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            batched_input = self._prepare_batch_input(batch, batch_size)
            gt_masks = batch['gt_mask'].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Forward Pass
                outputs = self.model(batched_input, multimask_output=True)
                
                total_loss = 0
                loss_dict_accum = {"total": 0, "bce": 0, "dice": 0, "iou_mse": 0}
                
                # 用來存視覺化用的 Tensor (只存第一個 Batch)
                first_batch_logits = None 

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

                    if i == 0:
                        first_batch_logits = full_res_logits
                    
                    # 計算 Loss (使用 Min-Loss Strategy)
                    sample_loss, sample_dict = self.criterion(
                        pred_masks=full_res_logits,
                        gt_mask=gt_masks[i],
                        iou_predictions=outputs[i]['iou_predictions'],
                        text_prompts=batched_input[i]['text_prompts']
                    )
                    
                    total_loss += sample_loss
                    for k, v in sample_dict.items():
                        loss_dict_accum[k] += v

                # 平均 Loss
                total_loss = total_loss / batch_size

            # Backward Pass
            self.scaler.scale(total_loss).backward()
            
            # [重要] 梯度裁減 (Gradient Clipping)
            # 必須在 unscale 之後，step 之前執行
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step_count += 1
            
            # 更新 Metric 顯示
            for k in epoch_metrics:
                val_to_add = loss_dict_accum[k]
                if torch.is_tensor(val_to_add):
                    val_to_add = val_to_add.item()
                epoch_metrics[k] += (val_to_add / batch_size)

            pbar.set_postfix(
                loss=total_loss.item(), 
                dice=(loss_dict_accum['dice']/batch_size),
                dice_score=1.0 - (loss_dict_accum['dice']/batch_size)
            )

            # -------------------------------------------------------
            # [修改 3] Debug 視覺化 (已修正 Bug)
            # -------------------------------------------------------
            if step_count % 1000 == 0 and first_batch_logits is not None:
                # 簡單取第一個 Batch, 第一個 Prompt, 第一個 Mask 來看
                # shape: (K, 3, 1024, 1024) -> 取 [0, 0] -> (1024, 1024)
                pred_logit = first_batch_logits[0, 0, :, :]
                mask_viz = torch.sigmoid(pred_logit).detach().cpu().numpy()
                
                # 存圖
                save_path = f"debug_viz/epoch_{epoch_index+1}_step_{step_count}.png"
                plt.imsave(save_path, mask_viz, cmap='gray')
                
                # 印出最大值，檢查是否全黑 (<0.1) 或全白 (>0.9)
                max_val = mask_viz.max()
                status = "🟢 OK" if max_val > 0.1 else "🔴 Collapsed (Black)"
                print(f"   📸 Snapshot saved! Max Value: {max_val:.4f} [{status}]")

        # 計算整個 Epoch 的平均指標
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
        
        # 根據 Validation Loss 調整 LR
        self.scheduler.step(avg_metrics['total'])
        
        return avg_metrics

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Checkpoint saved to {path}")