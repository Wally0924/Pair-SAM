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
from utils.new_loss import ContextLoss, MaskLoss

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

class AttentionMonitor:
    """
    監控 CMAAlignment 的訓練狀態。

    指標說明：
        last_conf_mean   : float [0, 1]
            UAWarpC 置信度全局均值。越高代表整體對齊品質越好。
            接近 0 → flow 估計不可靠（視角差異過大或 VGG 特徵無法匹配）。

        last_valid_ratio : float [0, 1]
            warp 後邊界內有效像素比例。接近 1 → flow 不超出影像邊界。
    """

    def __init__(self, fusion_module):
        self.last_conf_mean: float = 0.0
        self.last_valid_ratio: float = 0.0
        self.last_confidence_map = None   # (B,1,H,W) CPU tensor; updated each forward
        self.last_flow = None             # (B,2,H,W) CPU tensor; pixel displacement
        self._fusion_handle = None
        self._install(fusion_module)

    def _install(self, fusion_module):
        monitor = self

        def fusion_out_hook(module, _inputs, _output):
            monitor.last_conf_mean      = getattr(module, "_last_conf_mean",      0.0)
            monitor.last_valid_ratio    = getattr(module, "_last_valid_ratio",    0.0)
            monitor.last_confidence_map = getattr(module, "_last_confidence_map", None)
            monitor.last_flow           = getattr(module, "_last_flow",           None)

        self._fusion_handle = fusion_module.register_forward_hook(fusion_out_hook)

    def remove(self):
        if self._fusion_handle is not None:
            self._fusion_handle.remove()
            self._fusion_handle = None


class WeatherSAMTrainer:
    """
    負責訓練 WeatherSAM 的訓練器（Mask2Former-style），實作解耦式的多重 Loss 計算：
    包含 MaskLoss (Focal+Dice)、ContextLoss (CE)。
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
        focal_w = getattr(args, 'focal_weight', 3.0)
        dice_w = getattr(args, 'dice_weight', 1.0)
        # iou_w = getattr(args, 'iou_weight', 1.0)  # [Mask2Former] IoU MSE Loss 已移除
        decoder_lr_scale = getattr(args, 'decoder_lr_scale', 0.1)
        transformer_lr_scale = getattr(args, 'transformer_lr_scale', 0.01)
        
        print(f"📉 Initializing Decoupled Losses (CE: {ce_w}, Mask[Focal: {focal_w}, Dice: {dice_w}])")
        print(f"🔓 MaskDecoder LR scale: {decoder_lr_scale} (LR = {lr * decoder_lr_scale:.2e})")
        print(f"🔓 MaskDecoder Transformer LR scale: {transformer_lr_scale} (LR = {lr * transformer_lr_scale:.2e})")
        self.context_loss_fn = ContextLoss(ce_weight=ce_w).to(self.device)
        self.mask_loss_fn = MaskLoss(focal_weight=focal_w, dice_weight=dice_w)
        
        self.scaler = torch.amp.GradScaler('cuda', init_scale=8192)
        
        # 凍結與解凍策略
        for param in self.model.parameters():
            param.requires_grad = False
            
        # ── 主幹適配模組 (main LR) ──
        # 注意：self.model.fusion_module (CMAAlignment) 已在其 __init__ 中將 vgg_backbone
        # 與 warp_head 全部凍結（CMA pretrained UAWarpC），且本身無可訓練參數，故不列入。
        # gated_fusion (FlowGuidedSemanticAlignment) 的 cross_attn + gate_net + norm
        # 屬於核心可訓練模組，列入 main LR group。
        main_lr_modules = [
            self.model.condition_encoder,
            self.model.text_encoder.projection,
            self.model.context_fusion_head,
            self.model.gated_fusion,
        ]

        # ── MaskDecoder 新增/重訓模組 (decoder LR) ──
        # 這些模組直接產生 class mask logits，使用 decoder_lr_scale 與 main LR 解耦。
        decoder_lr_modules = [
            self.model.mask_decoder.output_upscaling,
            self.model.mask_decoder.class_mask_tokens,
            self.model.mask_decoder.class_hypernetworks_mlps,
        ]

        # ── 解凍 MaskDecoder Transformer (極低 LR) ──
        # transformer 在 SAM clean-image feature 上預訓練，fused feature 分布已偏移
        # 以極低 LR (1/100) fine-tune，讓 transformer 適應 weather fused feature 的分布
        decoder_transformer_modules = [
            self.model.mask_decoder.transformer,
        ]

        for module in main_lr_modules:
            for param in module.parameters():
                param.requires_grad = True

        for module in decoder_lr_modules:
            for param in module.parameters():
                param.requires_grad = True

        for module in decoder_transformer_modules:
            for param in module.parameters():
                param.requires_grad = True

        # ── 防禦性 re-freeze：確保 CMAAlignment 的 VGG + UAWarpC 永不被誤更新 ──
        for param in self.model.fusion_module.parameters():
            param.requires_grad = False

        self.model.pe_layer.requires_grad = True

        # ── 建立分離 LR 的 parameter groups ──
        main_params       = [p for m in main_lr_modules             for p in m.parameters() if p.requires_grad]
        decoder_params    = [p for m in decoder_lr_modules          for p in m.parameters() if p.requires_grad]
        decoder_tf_params = [p for m in decoder_transformer_modules for p in m.parameters() if p.requires_grad]
        pe_params         = [self.model.pe_layer] if self.model.pe_layer.requires_grad else []

        param_groups = [
            {'params': main_params,       'lr': lr,                        'name': 'main'},
            {'params': decoder_params,    'lr': lr * decoder_lr_scale,      'name': 'decoder'},
            {'params': decoder_tf_params, 'lr': lr * transformer_lr_scale, 'name': 'decoder_transformer'},
            {'params': pe_params,         'lr': lr,                        'name': 'pe_layer'},
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
        warmup_epochs = getattr(args, 'warmup_epochs', 5)

        def lr_lambda(epoch_idx):
            # 1. Warmup 階段: 線性上升
            if epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)
            
            # 2. Cosine Decay 階段: 緩慢下降
            else:
                # 計算進度 (0.0 ~ 1.0)
                progress = float(epoch_idx - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
                # 餘弦公式: 0.5 * (1 + cos(pi * progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        os.makedirs("debug_viz", exist_ok=True)

        # Attention 診斷監控器（CMAAlignment confidence + valid ratio）
        self.attn_monitor = AttentionMonitor(self.model.fusion_module)

    def _prepare_batch_input(self, batch, batch_size):
        batched_input = []
        use_cached_features = 'image_embedding' in batch
        
        for i in range(batch_size):
            input_dict = {
                'text_prompts': batch['text_prompts'][i],
                'original_size': batch['original_size'][i],
                'condition_id': batch['condition_id'][i].to(self.device),
            }
            if use_cached_features:
                input_dict['image_embedding'] = batch['image_embedding'][i].to(self.device)
            else:
                input_dict['image'] = batch['image'][i].to(self.device)
            # [CMAAlignment] 原始影像（cache 模式下 dataloader 已補充）
            if 'image' in batch:
                input_dict['image'] = batch['image'][i].to(self.device)
            if 'clear_image' in batch:
                input_dict['clear_image'] = batch['clear_image'][i].to(self.device)
            # [image-pair] clear-weather ViT-H embedding for CMAAlignment
            input_dict['clear_embedding'] = batch['clear_embedding'][i].to(self.device)
            batched_input.append(input_dict)
        return batched_input

    def train_epoch(self, epoch_index: int):
        """
        執行單一 Epoch 的模型訓練（Mask2Former-style 解耦 Multi-Loss）：
        Stage 1: MaskLoss (Focal+Dice，每類別 1 mask)。
        Stage 3: 漸進式梯度解放（前 5 epoch detach）。
        Stage 4: 組裝 19 個 class channel。
        Stage 5: ResidualDWConvFusion + ContextLoss (CE)。
        """
        # ★ 針對剛解開 detach 的 epoch (epoch_index == 5)，重置 Mask Decoder 的 Adam 動量
        # [Mask2Former] 重置清單已更新為新架構的模組，舊的 shared token 已不在梯度路徑中
        if epoch_index == 5:
            print(f"\n[INFO] Epoch 5: Clearing Adam state for MaskDecoder to prevent Momentum shock...")
            modules_to_reset = [
                self.model.mask_decoder.class_mask_tokens,
                self.model.mask_decoder.class_hypernetworks_mlps,
                self.model.mask_decoder.output_upscaling,
                self.model.mask_decoder.transformer,  # epoch 5 解除 detach 後 CE 梯度首次流入 transformer，重置動量避免 loss 跳升
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
            "conf_mean":        AverageMeter(),  # UAWarpC 置信度全局均值 [0, 1]
            "valid_ratio":      AverageMeter(),  # warp 邊界內有效像素比例 [0, 1]
            "alpha_mean":       AverageMeter(),  # gated_fusion alpha gate 均值（實際融合強度）
            "alpha_std":        AverageMeter(),  # alpha 空間變異，低=整張圖幾乎同一融合比例
            "alpha_high_ratio": AverageMeter(),  # alpha > 0.5 的像素比例
            "fusion_cos_sim":   AverageMeter(),  # f_curr vs f_ref_warped cosine similarity
            "pred_conf_mean":   AverageMeter(),  # 19-class softmax max probability
            "pred_entropy":     AverageMeter(),  # 19-class normalized entropy，低=分類更果斷
            "logit_margin":     AverageMeter(),  # top1-top2 probability margin，低=argmax 易破碎
            "grad_norm_main":   AverageMeter(),  # main param group L2 gradient norm（unscale 後）
            "grad_norm_decoder":AverageMeter(),  # decoder head gradient norm
            "scaler_scale":     AverageMeter(),  # AMP GradScaler 當前 scale factor
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
                outputs = self.model(batched_input)

                total_loss = torch.tensor(0.0, device=self.device)
                sample_ce_sum = 0.0
                sample_focal_sum = 0.0
                sample_dice_sum = 0.0
                sample_pred_conf_sum = 0.0
                sample_pred_entropy_sum = 0.0
                sample_margin_sum = 0.0
                first_batch_logits = None
                first_batch_gt = None
                first_batch_image = None
                first_batch_per_class_logits = None   # (K, 256, 256) before ContextFusionHead
                first_batch_class_ids = None          # List[int], active classes
                
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
                    acc_focal, acc_dice = 0.0, 0.0

                    # ── Stage 1: MaskLoss（每類別 1 個 mask，class_ids_out 固定為 [0..18]）──
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

                        mask_total_loss, focal, dice = self.mask_loss_fn(
                            low_res_logits_upscaled, target_masks_k, valid_mask_k
                        )
                        min_mask_loss = mask_total_loss.squeeze(1)   # (K,)
                        sample_total_loss += min_mask_loss.mean()
                        acc_focal = focal.squeeze(1).mean().item()
                        acc_dice  = dice.squeeze(1).mean().item()

                    # ── Stage 3: 前 5 epoch detach，防止初期 Focal Loss 梯度爆炸 ──
                    if num_prompts > 0:
                        if epoch_index < 5:
                            selected_logits = low_res_logits.detach()  # (K, 256, 256)
                        else:
                            selected_logits = low_res_logits           # (K, 256, 256)
                    else:
                        selected_logits = torch.empty((0, 256, 256), device=self.device)

                    # ── Stage 4: 所有 19 類別固定存在，直接使用（不需 -10 填充）──
                    full_class_logits = selected_logits.unsqueeze(0)  # (1, 19, 256, 256)
                    
                    # ── Stage 5: ContextFusionHead & Context Loss (CE) ──
                    fused_logits = self.model.context_fusion_head(full_class_logits)

                    with torch.no_grad():
                        probs_lr = torch.softmax(fused_logits.float(), dim=1)
                        top2 = torch.topk(probs_lr, k=2, dim=1).values
                        pred_conf = top2[:, 0].mean().item()
                        margin = (top2[:, 0] - top2[:, 1]).mean().item()
                        entropy = -(probs_lr * probs_lr.clamp_min(1e-8).log()).sum(dim=1)
                        entropy = (entropy / math.log(float(self.model.num_classes))).mean().item()
                    sample_pred_conf_sum += pred_conf
                    sample_margin_sum += margin
                    sample_pred_entropy_sum += entropy
                    
                    fused_logits_hr = self.model.postprocess_masks(
                        fused_logits,
                        input_size=(1024, 1024),
                        original_size=(1024, 1024),
                    )
                    
                    context_loss, ce_val = self.context_loss_fn(fused_logits_hr, gt_mask_i)
                    sample_total_loss += context_loss

                    if i == 0:
                        first_batch_logits = fused_logits.squeeze(0)
                        first_batch_gt = gt_mask_i.squeeze(0).detach()
                        first_batch_per_class_logits = selected_logits.detach().cpu()  # (K, 256, 256)
                        first_batch_class_ids = list(class_ids_out)
                        if 'image' in batch:
                            first_batch_image = batch['image'][0].detach()
                    
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

                # ── 需求 2: Per-group gradient norm（unscale 後、clip 前，反映真實梯度大小）──
                for g in self.optimizer.param_groups:
                    params_with_grad = [p for p in g['params'] if p.grad is not None]
                    if params_with_grad:
                        gn = torch.norm(
                            torch.stack([p.grad.detach().norm(2) for p in params_with_grad])
                        ).item()
                    else:
                        gn = 0.0
                    name = g.get('name', '')
                    if name == 'main':
                        losses['grad_norm_main'].update(gn)
                    elif name == 'decoder':
                        losses['grad_norm_decoder'].update(gn)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # ── 需求 3: AMP scaler scale factor（update 後讀取，下一步的縮放倍率）──
                losses['scaler_scale'].update(self.scaler.get_scale())
                self.optimizer.zero_grad()
            
            losses['total'].update(float(total_loss.item()), batch_size)
            losses['ce'].update(float(sample_ce_sum) / float(batch_size), batch_size)
            losses['focal'].update(float(sample_focal_sum) / float(batch_size), batch_size)
            losses['dice'].update(float(sample_dice_sum) / float(batch_size), batch_size)
            losses['conf_mean'].update(self.attn_monitor.last_conf_mean, batch_size)
            losses['valid_ratio'].update(self.attn_monitor.last_valid_ratio, batch_size)
            _last_alpha = getattr(self.model.gated_fusion, '_last_alpha', None)
            alpha_mean_val = float(_last_alpha.mean().item()) if _last_alpha is not None else 0.0
            losses['alpha_mean'].update(alpha_mean_val, batch_size)
            if _last_alpha is not None:
                losses['alpha_std'].update(float(_last_alpha.std().item()), batch_size)
                losses['alpha_high_ratio'].update(float((_last_alpha > 0.5).float().mean().item()), batch_size)
            _last_cos = getattr(self.model.gated_fusion, '_last_cos_sim', None)
            if _last_cos is not None:
                losses['fusion_cos_sim'].update(float(_last_cos.mean().item()), batch_size)
            losses['pred_conf_mean'].update(float(sample_pred_conf_sum) / float(batch_size), batch_size)
            losses['pred_entropy'].update(float(sample_pred_entropy_sum) / float(batch_size), batch_size)
            losses['logit_margin'].update(float(sample_margin_sum) / float(batch_size), batch_size)

            pbar.set_postfix(
                loss=f"{losses['total'].avg:.4f}",
                ce=f"{losses['ce'].avg:.4f}",
                focal=f"{losses['focal'].avg:.4f}",
                dice=f"{losses['dice'].avg:.4f}",
                ent=f"{losses['pred_entropy'].avg:.3f}",
                margin=f"{losses['logit_margin'].avg:.3f}",
                conf=f"{losses['conf_mean'].avg:.3f}",
                valid=f"{losses['valid_ratio'].avg:.3f}",
                alpha=f"{losses['alpha_mean'].avg:.4f}",
                gn_m=f"{losses['grad_norm_main'].avg:.3f}",
                gn_d=f"{losses['grad_norm_decoder'].avg:.3f}",
                scale=f"{losses['scaler_scale'].avg:.0f}",
            )

            if step_count > 0 and step_count % 500 == 0:
                if first_batch_logits is not None:
                    self._save_debug_snapshot(
                        first_batch_logits,
                        epoch_index,
                        step_count,
                        gt_mask=first_batch_gt,
                        image=first_batch_image,
                    )
                if first_batch_per_class_logits is not None and first_batch_class_ids:
                    self._save_per_class_decoder_viz(
                        first_batch_per_class_logits,
                        first_batch_class_ids,
                        epoch_index,
                        step_count,
                        image=first_batch_image,
                        gt_mask=first_batch_gt,
                    )
                self._save_fusion_quality_viz(batch, epoch_index, step_count)
        
        self.scheduler.step()
        avg_metrics = {k: v.avg for k, v in losses.items()}
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"   🔄 Learning Rate Updated: {current_lr:.2e}")
        
        return avg_metrics

    def _save_debug_snapshot(self, logits, epoch, step, gt_mask=None, image=None):
        """
        儲存訓練中間狀態的語意分割快照。
        顯示原圖、Cityscapes palette 預測、GT 與 max-softmax confidence。
        """
        num_classes = logits.shape[0]
        palette = torch.tensor([
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32],
        ], dtype=torch.float32)

        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=0)           # (C, H, W)
            pred  = probs.argmax(dim=0).cpu()                      # (H, W)
            conf  = probs.max(dim=0).values.cpu().numpy()          # (H, W) — max-softmax

        H, W = pred.shape
        pred_rgb = palette[pred.clamp(0, num_classes - 1)].numpy().astype(np.uint8)

        n_panels = 2
        if gt_mask is not None:
            n_panels += 1
        if image is not None:
            n_panels += 1

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        ax_idx = 0

        if image is not None:
            img_np = image.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            img_np = np.clip(img_np, 0, 255)
            axes[ax_idx].imshow(img_np)
            axes[ax_idx].set_title('Input Image', fontsize=9)
            axes[ax_idx].axis('off')
            ax_idx += 1

        axes[ax_idx].imshow(pred_rgb)
        axes[ax_idx].set_title('Prediction (ACDC/Cityscapes 19-class)', fontsize=9)
        axes[ax_idx].axis('off')
        ax_idx += 1

        if gt_mask is not None:
            gt_np = gt_mask.cpu().numpy()
            ignore_mask = (gt_np == 255)
            gt_vis = np.where(ignore_mask, 0, gt_np).astype(np.int64)
            gt_rgb = palette[np.clip(gt_vis, 0, num_classes - 1)].numpy().astype(np.uint8)
            gt_rgb[ignore_mask] = 0  # render ignore pixels as black instead of road color
            axes[ax_idx].imshow(gt_rgb)
            axes[ax_idx].set_title('Ground Truth', fontsize=9)
            axes[ax_idx].axis('off')
            ax_idx += 1

        im = axes[ax_idx].imshow(conf, cmap='plasma', vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
        axes[ax_idx].set_title('Max-Softmax Confidence', fontsize=9)
        axes[ax_idx].axis('off')

        plt.suptitle(
            f'Segmentation Snapshot  |  Epoch {epoch+1}  Step {step}  |  '
            f'conf_mean={conf.mean():.3f}',
            fontsize=9,
        )
        plt.tight_layout()
        os.makedirs('debug_viz/segmentation', exist_ok=True)
        plt.savefig(
            f'debug_viz/segmentation/epoch_{epoch+1}_step_{step:05d}.png',
            dpi=90, bbox_inches='tight',
        )
        plt.close(fig)

    def _save_per_class_decoder_viz(self, per_class_logits, class_ids, epoch, step,
                                     image=None, gt_mask=None):
        """
        顯示 mask decoder 原始輸出（ContextFusionHead 前）的每個 active class sigmoid heatmap。

        儲存格式：
          debug_viz/per_class/epoch_{E}_step_{S:05d}.png
          Grid layout：[輸入影像（選用）] [GT（選用）] [各 active class sigmoid 熱力圖 × K]
        """
        _CLASS_NAMES = [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        _PALETTE = torch.tensor([
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32],
        ], dtype=torch.float32)

        K = len(class_ids)
        if K == 0:
            return

        with torch.no_grad():
            sigmoids = torch.sigmoid(per_class_logits.float())  # (K, 256, 256)

        # --- 面板數量 ---
        n_extra = (1 if image is not None else 0) + (1 if gt_mask is not None else 0)
        n_panels = K + n_extra
        ncols = 5
        nrows = math.ceil(n_panels / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows),
                                 squeeze=False)
        axes_flat = axes.reshape(-1)
        ax_idx = 0

        # --- 輸入影像 ---
        if image is not None:
            img_np = image.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            img_np = np.clip(img_np, 0, 255)
            axes_flat[ax_idx].imshow(img_np)
            axes_flat[ax_idx].set_title('Input Image', fontsize=8)
            axes_flat[ax_idx].axis('off')
            ax_idx += 1

        # --- GT palette ---
        if gt_mask is not None:
            gt_np = gt_mask.cpu().numpy()
            ignore_mask = (gt_np == 255)
            gt_vis = np.where(ignore_mask, 0, gt_np).astype(np.int64)
            gt_rgb = _PALETTE[np.clip(gt_vis, 0, 18)].numpy().astype(np.uint8)
            gt_rgb[ignore_mask] = 0
            axes_flat[ax_idx].imshow(gt_rgb)
            axes_flat[ax_idx].set_title('Ground Truth', fontsize=8)
            axes_flat[ax_idx].axis('off')
            ax_idx += 1

        # --- 每個 active class 的 sigmoid heatmap ---
        for k, cls_id in enumerate(class_ids):
            prob = sigmoids[k].numpy()
            im = axes_flat[ax_idx].imshow(prob, cmap='RdYlGn', vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=axes_flat[ax_idx], fraction=0.04, pad=0.02)
            cls_name = _CLASS_NAMES[cls_id] if cls_id < len(_CLASS_NAMES) else f"cls{cls_id}"
            axes_flat[ax_idx].set_title(
                f'{cls_name}  (id={cls_id})\n'
                f'max={prob.max():.2f}  mean={prob.mean():.3f}',
                fontsize=7,
            )
            axes_flat[ax_idx].axis('off')
            ax_idx += 1

        # 隱藏多餘格子
        for j in range(ax_idx, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.suptitle(
            f'Decoder Output per Class (before ContextFusionHead)  |  '
            f'Epoch {epoch + 1}  Step {step}  |  {K} active classes',
            fontsize=9,
        )
        plt.tight_layout()
        os.makedirs('debug_viz/per_class', exist_ok=True)
        plt.savefig(
            f'debug_viz/per_class/epoch_{epoch + 1}_step_{step:05d}.png',
            dpi=90, bbox_inches='tight',
        )
        plt.close(fig)

    def _save_fusion_quality_viz(self, batch, epoch, step, sample_idx=0):
        """
        Fusion 品質評估可視化（4 欄）。

        [惡劣天氣原圖] [Cosine Sim f_curr vs f_ref_warped] [Alpha Gate] [晴天參考圖]

        設計意義：
          Cosine Sim → 幾何對齊後語意相似度，高 = warp 準確，負 = 對齊方向錯誤
          Alpha Gate → 高 = 大量注入晴天特徵；低 = 依賴自身特徵（信心不足位置為 0）
        """
        gf = self.model.gated_fusion

        cos_sim   = getattr(gf, '_last_cos_sim',  None)  # [B, H, W]
        alpha_map = getattr(gf, '_last_alpha',     None)  # [B, 1, H, W]

        if any(x is None for x in [cos_sim, alpha_map]):
            return

        img_curr = batch.get('image')
        img_ref  = batch.get('clear_image')
        if img_curr is None or img_ref is None:
            return

        b_idx = min(sample_idx, cos_sim.shape[0] - 1)
        VIZ   = 320

        # ── 1. 原始影像 ──
        def to_np(t):
            return F.interpolate(t[b_idx].float().unsqueeze(0) / 255.0,
                                 (VIZ, VIZ), mode='bilinear', align_corners=False
                                 ).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)

        curr_np = to_np(img_curr)
        ref_np  = to_np(img_ref)

        # ── 2. Cosine Similarity [H, W] → VIZ ──
        cos_np = F.interpolate(
            cos_sim[b_idx].unsqueeze(0).unsqueeze(0),
            (VIZ, VIZ), mode='bilinear', align_corners=False,
        ).squeeze().numpy()

        # ── 3. Alpha Gate [1, H, W] → VIZ ──
        alpha_np = F.interpolate(
            alpha_map[b_idx].unsqueeze(0),
            (VIZ, VIZ), mode='bilinear', align_corners=False,
        ).squeeze().numpy()

        # ── 4. 繪圖 ──
        panels = [
            (curr_np,   None,      'adverse weather',                      'gray'),
            (cos_np,    (-1, 1),   'cosine sim\n(f_curr vs f_ref_warped)', 'RdYlGn'),
            (alpha_np,  (0, 1),    'alpha gate\n(high=use clear ref)',      'hot'),
            (ref_np,    None,      'clear reference',                       'gray'),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for ax, (data, vrange, title, cmap) in zip(axes, panels):
            if data.ndim == 3:          # RGB 影像
                ax.imshow(data)
            else:
                vmin, vmax = vrange if vrange else (data.min(), data.max())
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

        plt.suptitle(
            f'Fusion Quality  |  Epoch {epoch+1}  Step {step}  |  '
            f'cos_sim_mean={cos_np.mean():.3f}  '
            f'alpha_mean={alpha_np.mean():.3f}',
            fontsize=9,
        )
        plt.tight_layout()
        os.makedirs('debug_viz/fusion', exist_ok=True)
        plt.savefig(
            f'debug_viz/fusion/epoch_{epoch+1}_step_{step:05d}.png',
            dpi=90, bbox_inches='tight',
        )
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
            "conf_mean": AverageMeter(),
            "valid_ratio": AverageMeter(),
            "alpha_mean": AverageMeter(),
            "alpha_std": AverageMeter(),
            "alpha_high_ratio": AverageMeter(),
            "fusion_cos_sim": AverageMeter(),
            "pred_conf_mean": AverageMeter(),
            "pred_entropy": AverageMeter(),
            "logit_margin": AverageMeter(),
        }
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_index+1} [Val]")
        step_count = 0

        # ── 需求 1: mIoU 混淆矩陣（epoch 級別累積，batch 結束後統一計算）──
        num_classes = self.model.num_classes
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

        for batch in pbar:
            batch_size = len(batch['text_prompts'])
            batched_input = self._prepare_batch_input(batch, batch_size)
            gt_masks = batch['gt_mask'].to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model(batched_input)

                sample_ce_sum = 0.0
                sample_focal_sum = 0.0
                sample_dice_sum = 0.0
                sample_pred_conf_sum = 0.0
                sample_pred_entropy_sum = 0.0
                sample_margin_sum = 0.0
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
                    acc_focal, acc_dice = 0.0, 0.0

                    # ── Stage 1: MaskLoss（class_ids_out 固定為 [0..18]）──
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

                    # ── Stage 3 & 4: 所有 19 類別固定存在，直接使用（不需 -10 填充）──
                    if num_prompts > 0:
                        selected_logits = low_res_logits  # (K, 256, 256)
                    else:
                        selected_logits = torch.empty((0, 256, 256), device=self.device)

                    full_class_logits = selected_logits.unsqueeze(0)  # (1, 19, 256, 256)
                    
                    # ── Stage 5: ContextFusionHead & Context Loss ──
                    fused_logits = self.model.context_fusion_head(full_class_logits)

                    with torch.no_grad():
                        probs_lr = torch.softmax(fused_logits.float(), dim=1)
                        top2 = torch.topk(probs_lr, k=2, dim=1).values
                        pred_conf = top2[:, 0].mean().item()
                        margin = (top2[:, 0] - top2[:, 1]).mean().item()
                        entropy = -(probs_lr * probs_lr.clamp_min(1e-8).log()).sum(dim=1)
                        entropy = (entropy / math.log(float(self.model.num_classes))).mean().item()
                    sample_pred_conf_sum += pred_conf
                    sample_margin_sum += margin
                    sample_pred_entropy_sum += entropy
                    
                    fused_logits_hr = self.model.postprocess_masks(
                        fused_logits,
                        input_size=(1024, 1024),
                        original_size=(1024, 1024),
                    )
                    
                    context_loss, ce_val = self.context_loss_fn(fused_logits_hr, gt_mask_i)

                    # ── 需求 1: 混淆矩陣更新（argmax 預測 vs GT，排除 ignore_index=255）──
                    pred_cls = fused_logits_hr.argmax(dim=1).squeeze(0)  # (H, W)
                    gt_flat  = gt_mask_i.squeeze(0)                      # (H, W)
                    valid_px = gt_flat != 255
                    if valid_px.any():
                        p = pred_cls[valid_px].cpu().long()
                        g = gt_flat[valid_px].cpu().long()
                        confusion += torch.bincount(
                            g * num_classes + p, minlength=num_classes * num_classes
                        ).reshape(num_classes, num_classes)

                    sample_total_loss = sample_total_loss + context_loss

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
            losses['conf_mean'].update(self.attn_monitor.last_conf_mean, batch_size)
            losses['valid_ratio'].update(self.attn_monitor.last_valid_ratio, batch_size)
            _last_alpha = getattr(self.model.gated_fusion, '_last_alpha', None)
            alpha_mean_val = float(_last_alpha.mean().item()) if _last_alpha is not None else 0.0
            losses['alpha_mean'].update(alpha_mean_val, batch_size)
            if _last_alpha is not None:
                losses['alpha_std'].update(float(_last_alpha.std().item()), batch_size)
                losses['alpha_high_ratio'].update(float((_last_alpha > 0.5).float().mean().item()), batch_size)
            _last_cos = getattr(self.model.gated_fusion, '_last_cos_sim', None)
            if _last_cos is not None:
                losses['fusion_cos_sim'].update(float(_last_cos.mean().item()), batch_size)
            losses['pred_conf_mean'].update(float(sample_pred_conf_sum) / float(batch_size), batch_size)
            losses['pred_entropy'].update(float(sample_pred_entropy_sum) / float(batch_size), batch_size)
            losses['logit_margin'].update(float(sample_margin_sum) / float(batch_size), batch_size)

            pbar.set_postfix(
                loss=losses['total'].avg,
                ce=losses['ce'].avg,
                focal=losses['focal'].avg,
                dice=losses['dice'].avg,
                ent=f"{losses['pred_entropy'].avg:.3f}",
                margin=f"{losses['logit_margin'].avg:.3f}",
                conf=f"{losses['conf_mean'].avg:.3f}",
                valid=f"{losses['valid_ratio'].avg:.3f}",
                alpha=f"{losses['alpha_mean'].avg:.4f}",
            )

        avg_metrics = {k: v.avg for k, v in losses.items()}

        # ── 需求 1: 從混淆矩陣計算 mIoU（epoch 結束後一次計算，精確且無近似誤差）──
        tp   = confusion.diag().float()
        fp   = (confusion.sum(0) - confusion.diag()).float()
        fn   = (confusion.sum(1) - confusion.diag()).float()
        iou_per_class = tp / (tp + fp + fn + 1e-6)          # (19,) [0, 1]
        valid_cls_mask = confusion.sum(1) > 0                # 只計算 GT 出現過的類別
        miou = float(iou_per_class[valid_cls_mask].mean()) if valid_cls_mask.any() else 0.0
        avg_metrics['miou']          = miou
        avg_metrics['per_class_iou'] = iou_per_class.tolist()

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
