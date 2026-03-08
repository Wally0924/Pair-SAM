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
        
        # --- Old Independent Binary Loss Configuration (Commented Out) ---
        # f_w = args.focal_weight if args else 2.0
        # d_w = args.dice_weight if args else 2.0
        # i_w = args.iou_weight if args else 1.0
        # ls = args.label_smoothing if args else 0.1
        # from utils.new_loss import SAMLoss
        # self.criterion_old = SAMLoss(focal_weight=f_w, dice_weight=d_w, iou_weight=i_w, label_smoothing=ls)
        # -------------------------------------------------------------
        
        # [修改] 使用 Semantic CrossEntropy Loss
        # [修改] 使用 CombinedSemanticLoss
        lr = args.lr if args else 1e-4
        
        ce_w = getattr(args, 'ce_weight', 1.0)
        focal_w = getattr(args, 'focal_weight', 2.0)
        dice_w = getattr(args, 'dice_weight', 2.0)
        from utils.new_loss import CombinedSemanticLoss
        print(f"📉 Initializing CombinedSemanticLoss (CE: {ce_w}, Focal: {focal_w}, Dice: {dice_w})")
        self.criterion = CombinedSemanticLoss(ce_weight=ce_w, focal_weight=focal_w, dice_weight=dice_w)
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 凍結與解凍策略
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 根據我們討論的策略：凍結核心大腦，只訓練適配與融合模組
        trainable_modules = [
            self.model.fusion_module,
            self.model.gate_module,
            self.model.location_encoder.output_projection,
            self.model.text_encoder.projection,
            self.model.semantic_fusion_head,
            self.model.mask_encoder,
            self.model.mask_decoder.iou_prediction_head,  # 解凍 IoU Head
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
                return float(epoch_idx) / float(warmup_epochs)
            
            # 2. Cosine Decay 階段: 緩慢下降
            else:
                # 計算進度 (0.0 ~ 1.0)
                progress = float(epoch_idx - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
                # 餘弦公式: 0.5 * (1 + cos(pi * progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=3
        # )

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
        epoch_metrics = {"total": 0.0, "ce": 0.0, "focal": 0.0, "dice": 0.0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
        
        # 取得 max_norm 設定
        max_norm = getattr(self.args, 'max_norm', 1.0)
        accumulation_steps = getattr(self.args, 'accumulate_steps', 4)
        
        self.optimizer.zero_grad()
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            batched_input = self._prepare_batch_input(batch, batch_size)
            gt_masks = batch['gt_mask'].to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model(batched_input, multimask_output=True)
                
                total_loss = torch.tensor(0.0, device=self.device)
                loss_dict_accum = {"ce": 0.0, "focal": 0.0, "dice": 0.0}
                first_batch_logits = None 
                
                # 這裡原本是 iterated over batch_size
                # 在我們的新設計中，每個 i 代表一張圖，且 outputs[i]['masks'] 是 (19, 1024, 1024)
                for i in range(batch_size):
                    low_res_logits = outputs[i]['low_res_logits']   # (K, 3, 256, 256)
                    iou_preds = outputs[i]['iou_predictions']       # (K, 3)
                    num_prompts = len(batched_input[i]['text_prompts'])
                    
                    # ── Step 1: Temperature-Scaled Soft Weighted Sum (可微分) ──
                    iou_temp = getattr(self.args, 'iou_temp', 0.1)
                    weights = F.softmax(iou_preds / iou_temp, dim=1)       # (K, 3)
                    weights = weights.unsqueeze(-1).unsqueeze(-1)           # (K, 3, 1, 1)
                    selected_logits = (low_res_logits * weights).sum(dim=1) # (K, 256, 256) ✅ 可微分
                    
                    # ── Step 2: list + cat 保留梯度鏈 ──
                    prompt_to_cls = {}
                    for prompt_idx, class_name in enumerate(batched_input[i]['text_prompts']):
                        if class_name in self.train_loader.dataset.CLASS_MAP:
                            cls_id = self.train_loader.dataset.CLASS_MAP[class_name]
                            if cls_id < self.model.num_classes:
                                prompt_to_cls[cls_id] = prompt_idx
                    
                    class_channels = []
                    for cls_id in range(self.model.num_classes):
                        if cls_id in prompt_to_cls:
                            class_channels.append(selected_logits[prompt_to_cls[cls_id]].unsqueeze(0))
                        else:
                            class_channels.append(torch.full((1, 256, 256), -10.0, device=self.device))
                    
                    full_class_logits = torch.cat(class_channels, dim=0).unsqueeze(0)  # (1, 19, 256, 256)
                    
                    # ── Step 3: FusionHead 在 256×256（VRAM ÷ 16）──
                    fused_logits = self.model.semantic_fusion_head(full_class_logits)
                    
                    # ── Step 4: postprocess_masks 上採樣到 1024×1024 ──
                    fused_logits_hr = self.model.postprocess_masks(
                        fused_logits,
                        input_size=(1024, 1024),
                        original_size=(1024, 1024),
                    )
                    
                    if i == 0:
                        first_batch_logits = fused_logits.squeeze(0)
                    
                    # ── Step 5: Combined Loss ──
                    gt_mask = gt_masks[i].unsqueeze(0).long()
                    active_prompts = batched_input[i]['text_prompts']
                    sample_loss, metrics = self.criterion(fused_logits_hr, gt_mask, active_prompts)
                    
                    total_loss += sample_loss
                    loss_dict_accum['ce'] += metrics['ce']
                    loss_dict_accum['focal'] += metrics['focal']
                    loss_dict_accum['dice'] += metrics['dice']

                loss_dict_accum['total'] = total_loss.item()
                total_loss = total_loss / batch_size
                loss_to_backward = total_loss / accumulation_steps

            self.scaler.scale(loss_to_backward).backward()
            
            step_count += 1
            
            # Gradient Accumulation Step
            if step_count % accumulation_steps == 0 or step_count == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            for k in loss_dict_accum:
                val_to_add = loss_dict_accum[k]
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + (val_to_add / batch_size)

            pbar.set_postfix(
                loss=total_loss.item(), 
                ce=(loss_dict_accum['ce']/batch_size),
                focal=(loss_dict_accum['focal']/batch_size),
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
        # logits is (C, H, W) after squeeze(0) in train_epoch
        pred_logit = logits[0, :, :]
        mask_viz = torch.sigmoid(pred_logit).detach().cpu().numpy()
        save_path = f"debug_viz/epoch_{epoch+1}_step_{step}.png"
        plt.imsave(save_path, mask_viz, cmap='gray')
        max_val = mask_viz.max()
        status = "🟢 OK" if max_val > 0.1 else "🔴 Collapsed"
        # print(f"   📸 Snapshot saved! Max Value: {max_val:.4f} [{status}]")

    @torch.no_grad()
    def validate_epoch(self, epoch_index):
        self.model.eval()
        epoch_metrics = {"total": 0.0, "ce": 0.0, "focal": 0.0, "dice": 0.0}
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_index+1} [Val]")
        step_count = 0
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            batched_input = self._prepare_batch_input(batch, batch_size)
            gt_masks = batch['gt_mask'].to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model(batched_input, multimask_output=True)
                
                loss_dict_accum = {"ce": 0.0, "focal": 0.0, "dice": 0.0}
                total_loss = torch.tensor(0.0, device=self.device)
                
                for i in range(batch_size):
                    low_res_logits = outputs[i]['low_res_logits']   # (K, 3, 256, 256)
                    iou_preds = outputs[i]['iou_predictions']       # (K, 3)
                    num_prompts = len(batched_input[i]['text_prompts'])
                    
                    # ── Step 1: Temperature-Scaled Soft Weighted Sum ──
                    iou_temp = getattr(self.args, 'iou_temp', 0.1)
                    weights = F.softmax(iou_preds / iou_temp, dim=1)
                    weights = weights.unsqueeze(-1).unsqueeze(-1)
                    selected_logits = (low_res_logits * weights).sum(dim=1)
                    
                    # ── Step 2: list + cat ──
                    prompt_to_cls = {}
                    for prompt_idx, class_name in enumerate(batched_input[i]['text_prompts']):
                        if class_name in self.val_loader.dataset.CLASS_MAP:
                            cls_id = self.val_loader.dataset.CLASS_MAP[class_name]
                            if cls_id < self.model.num_classes:
                                prompt_to_cls[cls_id] = prompt_idx
                    
                    class_channels = []
                    for cls_id in range(self.model.num_classes):
                        if cls_id in prompt_to_cls:
                            class_channels.append(selected_logits[prompt_to_cls[cls_id]].unsqueeze(0))
                        else:
                            class_channels.append(torch.full((1, 256, 256), -10.0, device=self.device))
                    
                    full_class_logits = torch.cat(class_channels, dim=0).unsqueeze(0)
                    
                    # ── Step 3: FusionHead at 256² ──
                    fused_logits = self.model.semantic_fusion_head(full_class_logits)
                    
                    # ── Step 4: postprocess_masks 上採樣 ──
                    fused_logits_hr = self.model.postprocess_masks(
                        fused_logits,
                        input_size=(1024, 1024),
                        original_size=(1024, 1024),
                    )
                    
                    gt_mask = gt_masks[i].unsqueeze(0).long()
                    active_prompts = batched_input[i]['text_prompts']
                    sample_loss, metrics = self.criterion(fused_logits_hr, gt_mask, active_prompts)
                    
                    total_loss += sample_loss
                    loss_dict_accum['ce'] += metrics['ce']
                    loss_dict_accum['focal'] += metrics['focal']
                    loss_dict_accum['dice'] += metrics['dice']
            
            step_count += 1
            loss_dict_accum['total'] = total_loss.item()
            for k in loss_dict_accum:
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + (loss_dict_accum[k] / batch_size)
                
        avg_metrics = {k: v / step_count for k, v in epoch_metrics.items()}
        # self.scheduler.step(avg_metrics['total'])
        # self.scheduler.step()
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