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
import random

from segment_anything.modeling import WeatherSAM 
from utils.new_loss import ContextLoss, MaskLoss, calculate_true_iou, ActiveBoundaryLoss

class AverageMeter:
    """計算並儲存當前值與平均值。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WeatherSAMTrainer:
    """
    負責訓練 WeatherSAM 的訓練器，實作解耦式的多重 Loss 計算：
    包含 MaskLoss (Focal+Dice), IoU MSE Loss, 以及 ContextLoss (CE)。
    """
    def __init__(
        self, 
        model: WeatherSAM, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        args=None
    ):
        """
        初始化訓練器，設定模型、資料載入器、優化器、Scheduler 以及損失函數。
        """
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = args.device
        self.args = args
        
        # --- 損失函數初始化 ---
        lr = args.lr if args else 1e-4
        
        ce_w = getattr(args, 'ce_weight', 1.0)
        focal_w = getattr(args, 'focal_weight', 20.0)
        dice_w = getattr(args, 'dice_weight', 1.0)
        iou_w = getattr(args, 'iou_weight', 1.0)
        abl_w = getattr(args, 'abl_weight', 1.0)
        abl_start = getattr(args, 'abl_start_epoch', 5)
        
        print(f"📉 Initializing Decoupled Losses (CE: {ce_w}, Mask[Focal: {focal_w}, Dice: {dice_w}], IoU: {iou_w})")
        print(f"📉 Active Boundary Loss (Weight: {abl_w}, Start Epoch: {abl_start})")
        self.context_loss_fn = ContextLoss(ce_weight=ce_w)
        self.mask_loss_fn = MaskLoss(focal_weight=focal_w, dice_weight=dice_w)
        self.iou_mse_loss_fn = torch.nn.MSELoss()
        self.iou_weight = iou_w
        self.abl_loss_fn = ActiveBoundaryLoss()
        self.abl_weight = abl_w
        self.abl_start_epoch = abl_start
        
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
            self.model.context_fusion_head,
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
                return float(epoch_idx + 1) / float(warmup_epochs + 1)
            
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

    def train_epoch(self, epoch_index: int):
        """
        執行單一 Epoch 的模型訓練，並實作解耦的 Multi-Loss 更新機制：
        Stage 1: MaskLoss (候選遮罩形狀優化)。
        Stage 2: IoU MSE Loss (IoU 預測評分優化)。
        Stage 3: 最佳遮罩挑選。
        Stage 4: ContextFusionHead 空間融合。
        Stage 5: ContextLoss (全域互斥性優化)。
        """
        self.model.train()
        losses = {
            "total": AverageMeter(),
            "ce": AverageMeter(),
            "focal": AverageMeter(),
            "dice": AverageMeter(),
            "iou": AverageMeter(),
            "abl": AverageMeter()
        }
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]")
        
        step_count = 0
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
                sample_ce_sum = 0.0
                sample_focal_sum = 0.0
                sample_dice_sum = 0.0
                sample_iou_sum = 0.0
                sample_abl_sum = 0.0
                first_batch_logits = None 
                
                for i in range(batch_size):
                    low_res_logits = outputs[i]['low_res_logits']   # (K, 3, 256, 256)
                    iou_preds = outputs[i]['iou_predictions']       # (K, 3)
                    num_prompts = len(batched_input[i]['text_prompts'])
                    
                    gt_mask_i = gt_masks[i].unsqueeze(0).long() # (1, 1024, 1024)
                    valid_mask_i = (gt_mask_i != 255).float().unsqueeze(1) # (1, 1, 1024, 1024)
                    # ── 初始化損失與目標類別對照表 ──
                    sample_total_loss = torch.tensor(0.0, device=self.device)
                    prompt_to_cls = {}
                    for prompt_idx, class_name in enumerate(batched_input[i]['text_prompts']):
                        if class_name in self.train_loader.dataset.CLASS_MAP:
                            cls_id = self.train_loader.dataset.CLASS_MAP[class_name]
                            if cls_id < self.model.num_classes:
                                prompt_to_cls[cls_id] = prompt_idx

                    acc_focal, acc_dice = 0.0, 0.0
                    
                    # ── Stage 1 & 2: MaskLoss (Focal+Dice) & IoU MSE Loss ──
                    if num_prompts > 0:
                        # 內部上升取樣至 1024x1024 (僅供計算 Loss 使用)
                        low_res_logits_upscaled = self.model.postprocess_masks(
                            low_res_logits,
                            input_size=(1024, 1024),
                            original_size=(1024, 1024),
                        ) # (K, 3, 1024, 1024)
                        
                        target_masks_k = []
                        for prompt_idx in range(num_prompts):
                            cls_id = next((cid for cid, p_idx in prompt_to_cls.items() if p_idx == prompt_idx), None)
                            if cls_id is not None:
                                tgt = (gt_mask_i == cls_id).float().unsqueeze(1) # (1, 1, 1024, 1024)
                                target_masks_k.append(tgt)
                            else:
                                target_masks_k.append(torch.zeros((1, 1, 1024, 1024), device=self.device))
                                
                        target_masks_k = torch.cat(target_masks_k, dim=0) # (K, 1, 1024, 1024)
                        valid_mask_k = valid_mask_i.expand(num_prompts, -1, -1, -1) # (K, 1, 1024, 1024)
                        
                        # 計算 Mask Loss (回傳格式 BxK)
                        mask_total_loss, focal, dice = self.mask_loss_fn(low_res_logits_upscaled, target_masks_k, valid_mask_k)
                        
                        # [Stage 1] 保留 V7 架構，從 3 個 candidate 中挑選 Loss 最低的最佳預測作為梯度的依據
                        min_mask_loss, min_indices = torch.min(mask_total_loss, dim=1) # (K,)
                        
                        sample_total_loss += min_mask_loss.mean()
                        acc_focal = focal[torch.arange(num_prompts), min_indices].mean().item()
                        acc_dice = dice[torch.arange(num_prompts), min_indices].mean().item()
                        
                        # [Stage 2] 計算真實 IoU，並使用 MSE 監督 IoU 預測頭的打分精準度
                        with torch.no_grad():
                            true_iou = calculate_true_iou(low_res_logits_upscaled, target_masks_k, valid_mask_k) # (K, 3)
                        
                        iou_loss = self.iou_mse_loss_fn(iou_preds, true_iou)
                        sample_total_loss += self.iou_weight * iou_loss
                        sample_iou_sum += iou_loss.item()

                    # ── Stage 3: Argmax Selection for ContextFusionHead ──
                    if num_prompts > 0:
                        best_mask_indices = torch.argmax(iou_preds, dim=1) # (K,)
                        # ★ 真正的兩階段解耦：阻斷 ContextLoss 的梯度流回 Mask Decoder 等特徵抽取器，解決梯度互相打架與爆炸的問題
                        selected_logits = low_res_logits[torch.arange(num_prompts), best_mask_indices].detach() # (K, 256, 256)
                    else:
                        selected_logits = torch.empty((0, 256, 256), device=self.device)
                    
                    # ── Stage 4: list + cat 保留梯度鏈給 Fusion Head ──
                    class_channels = []
                    for cls_id in range(self.model.num_classes):
                        if cls_id in prompt_to_cls:
                            class_channels.append(selected_logits[prompt_to_cls[cls_id]].unsqueeze(0))
                        else:
                            class_channels.append(torch.full((1, 256, 256), -10.0, device=self.device))
                    
                    full_class_logits = torch.cat(class_channels, dim=0).unsqueeze(0)  # (1, 19, 256, 256)
                    
                    # ── Stage 5: ContextFusionHead & Context Loss (CE) ──
                    fused_logits = self.model.context_fusion_head(full_class_logits)
                    
                    fused_logits_hr = self.model.postprocess_masks(
                        fused_logits,
                        input_size=(1024, 1024),
                        original_size=(1024, 1024),
                    )
                    
                    context_loss, ce_val = self.context_loss_fn(fused_logits_hr, gt_mask_i)
                    sample_total_loss += context_loss
                    
                    # ── Stage 6: Active Boundary Loss (ABL) ──
                    abl_effective_weight = self.abl_weight if epoch_index >= self.abl_start_epoch else 0.0
                    if abl_effective_weight > 0:
                        abl_loss = self.abl_loss_fn(fused_logits_hr, gt_mask_i)
                        sample_total_loss += abl_effective_weight * abl_loss
                        sample_abl_sum += abl_loss.item()
                    
                    if i == 0:
                        first_batch_logits = fused_logits.squeeze(0)
                    
                    total_loss = total_loss + sample_total_loss
                    sample_ce_sum += ce_val
                    sample_focal_sum += float(acc_focal)
                    sample_dice_sum += float(acc_dice)

                total_loss = total_loss / float(batch_size)
                loss_to_backward = total_loss / float(accumulation_steps)

            self.scaler.scale(loss_to_backward).backward()
            
            step_count += 1
            
            # Gradient Accumulation Step
            is_accumulation_step = (step_count % accumulation_steps == 0)
            is_last_step = (step_count == len(self.train_loader))
            if is_accumulation_step or (is_last_step and not is_accumulation_step):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            losses['total'].update(float(total_loss.item()), batch_size)
            losses['ce'].update(float(sample_ce_sum) / float(batch_size), batch_size)
            losses['focal'].update(float(sample_focal_sum) / float(batch_size), batch_size)
            losses['dice'].update(float(sample_dice_sum) / float(batch_size), batch_size)
            losses['iou'].update(float(sample_iou_sum) / float(batch_size), batch_size)
            losses['abl'].update(float(sample_abl_sum) / float(batch_size), batch_size)

            pbar.set_postfix(
                loss=losses['total'].avg, 
                ce=losses['ce'].avg,
                focal=losses['focal'].avg,
                dice=losses['dice'].avg,
                iou=losses['iou'].avg,
                abl=losses['abl'].avg
            )

            if step_count % 1000 == 0 and first_batch_logits is not None:
                self._save_debug_snapshot(first_batch_logits, epoch_index, step_count)
        
        self.scheduler.step()
        avg_metrics = {k: v.avg for k, v in losses.items()}
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"   🔄 Learning Rate Updated: {current_lr:.2e}")
        
        return avg_metrics

    def _save_debug_snapshot(self, logits, epoch, step):
        # logits 已經是 (19, 256, 256) 的預測圖 (經過 squeeze(0))
        # 隨機挑選一個類別，避免每次都選到面積最大的類別 (e.g. road)
        num_classes = logits.shape[0]
        rand_class_idx = random.randint(0, num_classes - 1)
        
        # 截出隨機選到的類別的圖
        pred_logit = logits[rand_class_idx, :, :]
        mask_viz = torch.sigmoid(pred_logit).detach().cpu().numpy()
        
        # 存檔並在檔名標註這是哪一個類別
        cls_name = self.train_loader.dataset.ID_TO_NAME.get(rand_class_idx, f"unknown_{rand_class_idx}")
        save_path = f"debug_viz/epoch_{epoch+1}_step_{step}_cls_{rand_class_idx}_{cls_name}.png"
        plt.imsave(save_path, mask_viz, cmap='gray')

    @torch.no_grad()
    def validate_epoch(self, epoch_index: int):
        """
        執行單一 Epoch 的模型驗證，邏輯與 train_epoch 保持一致，但不計算梯度。
        """
        self.model.eval()
        losses = {
            "total": AverageMeter(),
            "ce": AverageMeter(),
            "focal": AverageMeter(),
            "dice": AverageMeter(),
            "iou": AverageMeter(),
            "abl": AverageMeter()
        }
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_index+1} [Val]")
        step_count = 0
        
        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            batched_input = self._prepare_batch_input(batch, batch_size)
            gt_masks = batch['gt_mask'].to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model(batched_input, multimask_output=True)
                
                sample_ce_sum = 0.0
                sample_focal_sum = 0.0
                sample_dice_sum = 0.0
                sample_iou_sum = 0.0
                sample_abl_sum = 0.0
                total_loss = torch.tensor(0.0, device=self.device)
                
                for i in range(batch_size):
                    low_res_logits = outputs[i]['low_res_logits']   # (K, 3, 256, 256)
                    iou_preds = outputs[i]['iou_predictions']       # (K, 3)
                    num_prompts = len(batched_input[i]['text_prompts'])
                    
                    gt_mask_i = gt_masks[i].unsqueeze(0).long()
                    valid_mask_i = (gt_mask_i != 255).float().unsqueeze(1)
                    
                    # ── 初始化損失與目標類別對照表 ──
                    sample_total_loss = torch.tensor(0.0, device=self.device)
                    prompt_to_cls = {}
                    for prompt_idx, class_name in enumerate(batched_input[i]['text_prompts']):
                        if class_name in self.val_loader.dataset.CLASS_MAP:
                            cls_id = self.val_loader.dataset.CLASS_MAP[class_name]
                            if cls_id < self.model.num_classes:
                                prompt_to_cls[cls_id] = prompt_idx

                    acc_focal, acc_dice = 0.0, 0.0
                    
                    # ── Stage 1 & 2: MaskLoss & IoU MSE Loss ──
                    if num_prompts > 0:
                        low_res_logits_upscaled = self.model.postprocess_masks(
                            low_res_logits,
                            input_size=(1024, 1024),
                            original_size=(1024, 1024),
                        )
                        
                        target_masks_k = []
                        for prompt_idx in range(num_prompts):
                            cls_id = next((cid for cid, p_idx in prompt_to_cls.items() if p_idx == prompt_idx), None)
                            if cls_id is not None:
                                tgt = (gt_mask_i == cls_id).float().unsqueeze(1)
                                target_masks_k.append(tgt)
                            else:
                                target_masks_k.append(torch.zeros((1, 1, 1024, 1024), device=self.device))
                                
                        target_masks_k = torch.cat(target_masks_k, dim=0)
                        valid_mask_k = valid_mask_i.expand(num_prompts, -1, -1, -1)
                        
                        mask_total_loss, focal, dice = self.mask_loss_fn(low_res_logits_upscaled, target_masks_k, valid_mask_k)
                        min_mask_loss, min_indices = torch.min(mask_total_loss, dim=1)
                        
                        sample_total_loss = sample_total_loss + min_mask_loss.mean()
                        acc_focal = float(focal[torch.arange(num_prompts), min_indices].mean().item())
                        acc_dice = float(dice[torch.arange(num_prompts), min_indices].mean().item())
                        
                        true_iou = calculate_true_iou(low_res_logits_upscaled, target_masks_k, valid_mask_k)
                        iou_loss = self.iou_mse_loss_fn(iou_preds, true_iou)
                        sample_total_loss = sample_total_loss + (self.iou_weight * iou_loss)
                        sample_iou_sum += iou_loss.item()

                    # ── Stage 3: Argmax Selection ──
                    if num_prompts > 0:
                        best_mask_indices = torch.argmax(iou_preds, dim=1)
                        selected_logits = low_res_logits[torch.arange(num_prompts), best_mask_indices].detach()
                    else:
                        selected_logits = torch.empty((0, 256, 256), device=self.device)
                    
                    # ── Stage 4: list + cat ──
                    class_channels = []
                    for cls_id in range(self.model.num_classes):
                        if cls_id in prompt_to_cls:
                            class_channels.append(selected_logits[prompt_to_cls[cls_id]].unsqueeze(0))
                        else:
                            class_channels.append(torch.full((1, 256, 256), -10.0, device=self.device))
                    
                    full_class_logits = torch.cat(class_channels, dim=0).unsqueeze(0)
                    
                    # ── Stage 5: ContextFusionHead & Context Loss ──
                    fused_logits = self.model.context_fusion_head(full_class_logits)
                    
                    fused_logits_hr = self.model.postprocess_masks(
                        fused_logits,
                        input_size=(1024, 1024),
                        original_size=(1024, 1024),
                    )
                    
                    context_loss, ce_val = self.context_loss_fn(fused_logits_hr, gt_mask_i)
                    sample_total_loss = sample_total_loss + context_loss
                    
                    # ── Stage 6: Active Boundary Loss (ABL) ──
                    abl_effective_weight = self.abl_weight if epoch_index >= self.abl_start_epoch else 0.0
                    if abl_effective_weight > 0:
                        abl_loss = self.abl_loss_fn(fused_logits_hr, gt_mask_i)
                        sample_total_loss = sample_total_loss + abl_effective_weight * abl_loss
                        sample_abl_sum += abl_loss.item()
                    
                    total_loss = total_loss + sample_total_loss
                    sample_ce_sum += float(ce_val)
                    sample_focal_sum += acc_focal
                    sample_dice_sum += acc_dice
            
            step_count += 1
            
            total_loss_avg = float(total_loss.item()) / float(batch_size)
            losses['total'].update(total_loss_avg, batch_size)
            losses['ce'].update(float(sample_ce_sum) / float(batch_size), batch_size)
            losses['focal'].update(float(sample_focal_sum) / float(batch_size), batch_size)
            losses['dice'].update(float(sample_dice_sum) / float(batch_size), batch_size)
            losses['iou'].update(float(sample_iou_sum) / float(batch_size), batch_size)
            losses['abl'].update(float(sample_abl_sum) / float(batch_size), batch_size)
            
            pbar.set_postfix(
                loss=losses['total'].avg, 
                ce=losses['ce'].avg,
                focal=losses['focal'].avg,
                dice=losses['dice'].avg,
                iou=losses['iou'].avg,
                abl=losses['abl'].avg
            )
                
        avg_metrics = {k: v.avg for k, v in losses.items()}
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