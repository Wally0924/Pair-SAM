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
from utils.new_loss import ContextLoss, MaskLoss, ActiveBoundaryLoss

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
    負責訓練 WeatherSAM 的訓練器（Mask2Former-style），實作解耦式的多重 Loss 計算：
    包含 MaskLoss (Focal+Dice)、ContextLoss (CE)、Active Boundary Loss (ABL)。
    # IoU MSE Loss 已移除：新架構每類別僅 1 mask，無候選選擇需求。
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
        # iou_w = getattr(args, 'iou_weight', 1.0)  # [Mask2Former] IoU MSE Loss 已移除
        abl_w = getattr(args, 'abl_weight', 1.0)
        abl_start = getattr(args, 'abl_start_epoch', 5)
        decoder_lr_scale = getattr(args, 'decoder_lr_scale', 0.1)
        transformer_lr_scale = getattr(args, 'transformer_lr_scale', 0.01)
        
        print(f"📉 Initializing Decoupled Losses (CE: {ce_w}, Mask[Focal: {focal_w}, Dice: {dice_w}])")
        print(f"📉 Active Boundary Loss (Weight: {abl_w}, Start Epoch: {abl_start})")
        print(f"🔓 MaskDecoder Transformer LR scale: {decoder_lr_scale} (LR = {lr * decoder_lr_scale:.2e})")
        self.context_loss_fn = ContextLoss(ce_weight=ce_w).to(self.device)
        self.mask_loss_fn = MaskLoss(focal_weight=focal_w, dice_weight=dice_w)
        self.abl_loss_fn = ActiveBoundaryLoss()
        self.abl_weight = abl_w
        self.abl_start_epoch = abl_start
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 凍結與解凍策略
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 根據我們討論的策略：凍結核心大腦，只訓練適配與融合模組
        # ── Location / Condition 模組選擇 ──
        # use_condition_embedding=True  → ACDC 模式：凍結 LocationEncoder，訓練 ConditionEncoder
        # use_condition_embedding=False → Cityscapes 模式：訓練 LocationEncoder.output_projection（原行為）
        use_condition_embedding = getattr(args, 'use_condition_embedding', False)
        if use_condition_embedding:
            print("🌦️  [Mode] ACDC fine-tune: LocationEncoder 凍結，啟用 ConditionEncoder")
            location_module = self.model.condition_encoder
        else:
            print("🗺️  [Mode] Cityscapes: 使用 LocationEncoder.output_projection")
            location_module = self.model.location_encoder.output_projection

        # ── 主幹適配模組 (main LR) ──
        main_lr_modules = [
            self.model.fusion_module,
            self.model.gate_module,
            location_module,
            self.model.text_encoder.projection,
            self.model.context_fusion_head,
            self.model.mask_encoder,
            # ── 原始 SAM decoder heads（保留向後兼容）──
            # ── [Mask2Former] 19 類別專屬模組 ──
            self.model.mask_decoder.output_upscaling,
            self.model.mask_decoder.class_mask_tokens,
            self.model.mask_decoder.class_hypernetworks_mlps,
        ]

        # ── 解凍 MaskDecoder Query Embeddings (低 LR) ──
        # iou_token (1×256) + mask_tokens (4×256) = ~1280 params，極輕量
        decoder_token_modules = [
            self.model.mask_decoder.iou_token,
            self.model.mask_decoder.mask_tokens,
        ]

        # ── 解凍 MaskDecoder Transformer (極低 LR) ──
        # 診斷確認 cross-attention 對 fused weather feature 近乎均勻分布（focus ratio < 0.04）
        # 原因：transformer 在 SAM clean-image feature 上預訓練，fused feature 分布已偏移
        # 以極低 LR (1/100) fine-tune，讓 transformer 適應 weather fused feature 的分布
        # 同時保持對 SAM pretrained prior 的破壞降到最低
        decoder_transformer_modules = [
            self.model.mask_decoder.transformer,
        ]

        for module in main_lr_modules:
            for param in module.parameters():
                param.requires_grad = True

        for module in decoder_token_modules:
            for param in module.parameters():
                param.requires_grad = True

        for module in decoder_transformer_modules:
            for param in module.parameters():
                param.requires_grad = True

        self.model.pe_layer.requires_grad = True

        # ── 建立分離 LR 的 parameter groups ──
        main_params        = [p for m in main_lr_modules            for p in m.parameters() if p.requires_grad]
        decoder_tok_params = [p for m in decoder_token_modules      for p in m.parameters() if p.requires_grad]
        decoder_tf_params  = [p for m in decoder_transformer_modules for p in m.parameters() if p.requires_grad]
        pe_params          = [self.model.pe_layer] if self.model.pe_layer.requires_grad else []

        param_groups = [
            {'params': main_params,        'lr': lr,                             'name': 'main'},
            {'params': decoder_tok_params, 'lr': lr * decoder_lr_scale,          'name': 'decoder_tokens'},
            {'params': decoder_tf_params,  'lr': lr * transformer_lr_scale,      'name': 'decoder_transformer'},
            {'params': pe_params,          'lr': lr,                             'name': 'pe_layer'},
        ]
        # 過濾掉空的 group
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        def count_params(params):
            return sum(p.numel() for p in params)

        total_trainable = sum(count_params(g['params']) for g in param_groups)
        print(f"✅ 總可訓練參數數量: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
        for g in param_groups:
            n = count_params(g['params'])
            print(f"   • [{g['name']}] {n:,} params ({n/1e6:.3f}M), LR={g['lr']:.2e}")

        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-2)

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
                'location': batch['location'][i].to(self.device),
                'condition_id': batch['condition_id'][i].to(self.device),
            }
            if use_cached_features:
                input_dict['image_embedding'] = batch['image_embedding'][i].to(self.device)
            else:
                input_dict['image'] = batch['image'][i].to(self.device)
            batched_input.append(input_dict)
        return batched_input

    def train_epoch(self, epoch_index: int):
        """
        執行單一 Epoch 的模型訓練（Mask2Former-style 解耦 Multi-Loss）：
        Stage 1: MaskLoss (Focal+Dice，每類別 1 mask)。
        Stage 3: 漸進式梯度解放（前 5 epoch detach）。
        Stage 4: 組裝 19 個 class channel。
        Stage 5: ResidualDWConvFusion + ContextLoss (CE)。
        Stage 6: Active Boundary Loss (ABL，延遲啟動)。
        """
        # ★ 針對剛解開 detach 的 epoch (epoch_index == 5)，重置 Mask Decoder 的 Adam 動量
        # [Mask2Former] 重置清單已更新為新架構的模組，舊的 shared token 已不在梯度路徑中
        if epoch_index == 5:
            print(f"\n[INFO] Epoch 5: Clearing Adam state for MaskDecoder to prevent Momentum shock...")
            modules_to_reset = [
                self.model.mask_decoder.class_mask_tokens,
                self.model.mask_decoder.class_hypernetworks_mlps,
                self.model.mask_decoder.output_upscaling,
            ]
            for module in modules_to_reset:
                for param in module.parameters():
                    if param in self.optimizer.state:
                        del self.optimizer.state[param]

        self.model.train()
        losses = {
            "total": AverageMeter(),
            "ce": AverageMeter(),
            "focal": AverageMeter(),
            "dice": AverageMeter(),
            # "iou": AverageMeter(),  # [Mask2Former] IoU MSE Loss 已移除，新架構每類別僅 1 mask，無候選選擇需求
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
                # sample_iou_sum = 0.0  # [Mask2Former] IoU MSE Loss 已移除
                sample_abl_sum = 0.0
                first_batch_logits = None
                
                for i in range(batch_size):
                    # [Mask2Former] 新輸出格式
                    low_res_logits = outputs[i]['low_res_logits'].squeeze(0)  # (K, 256, 256)
                    class_ids_out  = outputs[i]['class_ids']                  # List[int], len=K
                    num_prompts    = len(class_ids_out)

                    gt_mask_i = gt_masks[i].unsqueeze(0).long()  # (1, 1024, 1024)
                    if batch['invalid_mask'][i].any():
                        gt_mask_i = gt_mask_i.clone()
                        gt_mask_i[batch['invalid_mask'][i].to(self.device).unsqueeze(0)] = 255
                    valid_mask_i = (gt_mask_i != 255).float().unsqueeze(1)  # (1, 1, 1024, 1024)

                    sample_total_loss = torch.tensor(0.0, device=self.device)
                    # class_ids_out 已直接給出對應關係：index k → cls_id = class_ids_out[k]
                    prompt_to_cls = {cls_id: k for k, cls_id in enumerate(class_ids_out)}

                    acc_focal, acc_dice = 0.0, 0.0

                    # ── Stage 1: MaskLoss（Mask2Former 簡化版：每類別 1 個 mask，無 candidate 選擇）──
                    if num_prompts > 0:
                        # (K, 256, 256) → (K, 1, 256, 256) 供 postprocess 使用
                        low_res_logits_upscaled = self.model.postprocess_masks(
                            low_res_logits.unsqueeze(1),
                            input_size=(1024, 1024),
                            original_size=(1024, 1024),
                        )  # (K, 1, 1024, 1024)

                        target_masks_k = []
                        for cls_id in class_ids_out:
                            tgt = (gt_mask_i == cls_id).float().unsqueeze(1)  # (1, 1, 1024, 1024)
                            target_masks_k.append(tgt)
                        target_masks_k = torch.cat(target_masks_k, dim=0)     # (K, 1, 1024, 1024)
                        valid_mask_k   = valid_mask_i.expand(num_prompts, -1, -1, -1)

                        # 回傳 (K, 1) — 每類別 1 個 candidate，不需再選最佳
                        mask_total_loss, focal, dice = self.mask_loss_fn(
                            low_res_logits_upscaled, target_masks_k, valid_mask_k
                        )
                        min_mask_loss = mask_total_loss.squeeze(1)   # (K,)
                        sample_total_loss += min_mask_loss.mean()
                        acc_focal = focal.squeeze(1).mean().item()
                        acc_dice  = dice.squeeze(1).mean().item()

                    # ── Stage 3: 直接使用 mask（Mask2Former 無需 argmax 選 candidate）──
                    if num_prompts > 0:
                        # 前 5 epoch 漸進式梯度解放（防止初期 Focal Loss 梯度爆炸）
                        if epoch_index < 5:
                            selected_logits = low_res_logits.detach()  # (K, 256, 256)
                        else:
                            selected_logits = low_res_logits           # (K, 256, 256)
                    else:
                        selected_logits = torch.empty((0, 256, 256), device=self.device)

                    # ── Stage 4: 組裝 19 個 class channel ──
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
                    abl_effective_weight = self.abl_weight if epoch_index + 1 >= self.abl_start_epoch else 0.0
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
            # losses['iou'].update(float(sample_iou_sum) / float(batch_size), batch_size)  # [Mask2Former] 已移除
            losses['abl'].update(float(sample_abl_sum) / float(batch_size), batch_size)

            pbar.set_postfix(
                loss=losses['total'].avg,
                ce=losses['ce'].avg,
                focal=losses['focal'].avg,
                dice=losses['dice'].avg,
                # iou=losses['iou'].avg,  # [Mask2Former] 已移除
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
        """
        儲存訓練中間狀態的視覺化快照。
        以 softmax 機率顯示，避免 sigmoid(raw_logit) 飽和成全白的問題。
        """
        num_classes = logits.shape[0]
        rand_class_idx = random.randint(0, num_classes - 1)
        cls_name = self.train_loader.dataset.ID_TO_NAME.get(rand_class_idx, f"unknown_{rand_class_idx}")

        # Softmax 機率（正確的視覺化方式，值域 0~1 且加總為 1，不會飽和）
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=0)  # (19, H, W)

        # Panel 1: 隨機類別的 softmax 機率圖
        prob_map = probs[rand_class_idx].cpu().numpy()

        # Panel 2: argmax 預測圖（正規化到 0~1 用於灰度顯示）
        pred_class = torch.argmax(probs, dim=0).cpu().numpy()
        pred_norm = pred_class.astype(float) / max(num_classes - 1, 1)

        # Panel 3: 最大 softmax 機率（信心度圖）
        confidence = probs.max(dim=0).values.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 不要硬性指定 vmin/vmax，讓 matplotlib 自動基於 min/max 展開對比度
        im0 = axes[0].imshow(prob_map, cmap='magma')
        axes[0].set_title(f"P(cls={rand_class_idx} {cls_name})")
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pred_class, cmap='nipy_spectral') # 直接顯示 class idx，用高對比 cmap
        axes[1].set_title("Argmax Prediction")
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(confidence, cmap='viridis')
        axes[2].set_title("Confidence (max softmax)")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        save_path = f"debug_viz/epoch_{epoch+1}_step_{step}_cls_{rand_class_idx}_{cls_name}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=80)
        plt.close(fig)

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
            # "iou": AverageMeter(),  # [Mask2Former] IoU MSE Loss 已移除
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
                # sample_iou_sum = 0.0  # [Mask2Former] IoU MSE Loss 已移除
                sample_abl_sum = 0.0
                total_loss = torch.tensor(0.0, device=self.device)
                
                for i in range(batch_size):
                    # [Mask2Former] 新輸出格式
                    low_res_logits = outputs[i]['low_res_logits'].squeeze(0)  # (K, 256, 256)
                    class_ids_out  = outputs[i]['class_ids']                  # List[int], len=K
                    num_prompts    = len(class_ids_out)

                    gt_mask_i = gt_masks[i].unsqueeze(0).long()
                    if batch['invalid_mask'][i].any():
                        gt_mask_i = gt_mask_i.clone()
                        gt_mask_i[batch['invalid_mask'][i].to(self.device).unsqueeze(0)] = 255
                    valid_mask_i = (gt_mask_i != 255).float().unsqueeze(1)

                    sample_total_loss = torch.tensor(0.0, device=self.device)
                    prompt_to_cls = {cls_id: k for k, cls_id in enumerate(class_ids_out)}

                    acc_focal, acc_dice = 0.0, 0.0

                    # ── Stage 1: MaskLoss ──
                    if num_prompts > 0:
                        low_res_logits_upscaled = self.model.postprocess_masks(
                            low_res_logits.unsqueeze(1),
                            input_size=(1024, 1024),
                            original_size=(1024, 1024),
                        )  # (K, 1, 1024, 1024)

                        target_masks_k = []
                        for cls_id in class_ids_out:
                            tgt = (gt_mask_i == cls_id).float().unsqueeze(1)
                            target_masks_k.append(tgt)
                        target_masks_k = torch.cat(target_masks_k, dim=0)
                        valid_mask_k   = valid_mask_i.expand(num_prompts, -1, -1, -1)

                        mask_total_loss, focal, dice = self.mask_loss_fn(
                            low_res_logits_upscaled, target_masks_k, valid_mask_k
                        )
                        min_mask_loss = mask_total_loss.squeeze(1)
                        sample_total_loss = sample_total_loss + min_mask_loss.mean()
                        acc_focal = float(focal.squeeze(1).mean().item())
                        acc_dice  = float(dice.squeeze(1).mean().item())

                    # ── Stage 3: 直接使用 mask（無 candidate 選擇）──
                    if num_prompts > 0:
                        selected_logits = low_res_logits  # (K, 256, 256)
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
                    abl_effective_weight = self.abl_weight if epoch_index + 1 >= self.abl_start_epoch else 0.0
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
            # losses['iou'].update(float(sample_iou_sum) / float(batch_size), batch_size)  # [Mask2Former] 已移除
            losses['abl'].update(float(sample_abl_sum) / float(batch_size), batch_size)

            pbar.set_postfix(
                loss=losses['total'].avg,
                ce=losses['ce'].avg,
                focal=losses['focal'].avg,
                dice=losses['dice'].avg,
                # iou=losses['iou'].avg,  # [Mask2Former] 已移除
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