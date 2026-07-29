# pretrain_cityscapes.py
"""
Stage-1：在 Cityscapes 晴天影像上全量 fine-tune PairSAM 的 ViT-H encoder。

目標
    產出一個「域內化」的 SAM ViT-H encoder（sam_vit_h_cityscapes_encoder.pth），
    作為 Stage-2（ACDC 完整 PairSAM）唯一改動的 encoder 初始權重，取代原始
    SA-1B 的 sam_vit_h_4b8939.pth。Stage-2 訓練腳本 train.py 完全不需修改，
    只要把 --checkpoint 指向本腳本輸出的 encoder-only 權重即可。

pretrain 模型 = FULL 模型拔除 Adapter 與 condition embedding：
    build_pair_sam_from_config(use_vgg_adapter=False, cond=False)
    → 跳過 VGG adapter + CMA pre_align（不讀 clear_image），
      condition embedding 退化為常數（forward 內 use_cond=False 固定索引 0）。
    保留：ViT-H encoder + text prompt(19 類名) + prompt_encoder + mask_decoder
          + context_fusion_head，作為驅動 encoder 學習的「臨時頭」。

訓練策略
    - encoder 從 SA-1B 權重出發，全量解凍（patch_embed / pos_embed / 32 blocks / neck）。
    - 單卡 RTX 4090 / 24GB：gradient checkpointing 硬性開啟 + bf16 autocast
      + batch_size=1 + 梯度累積，避免 OOM。
    - Loss 重用 utils/new_loss.py 的 MaskLoss(Dice) + ContextLoss(CE，可選 Lovász)。
    - 每 eval_interval 個 epoch 以 tiling 推論在 Cityscapes val 算 mIoU，存 best。

只保存 encoder：
    checkpoint 僅含 image_encoder.* 權重，確保 Stage-2 build 的 state_dict 過濾
    只覆蓋 encoder、其餘模組維持隨機初始化 —— 對應「只改 encoder init」的乾淨 ablation。
"""
import os
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything.build_pair_sam import build_pair_sam_from_config
from utils.new_loss import (
    ContextLoss, MaskLoss, CITYSCAPES_CLASS_WEIGHTS, lovasz_weight_for_epoch,
)
from utils.seg_metrics import SegMetricsCalculator, CITYSCAPES_CLASSES
from utils.cityscapes_dataloader import CityscapesPretrainDataset, IGNORE_INDEX


# ---------------------------------------------------------------------------
# 參數
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Stage-1 Cityscapes encoder pretrain")
    # 資料
    p.add_argument("--images_root", type=str,
                   default="/home/rvl1421/Datasets/Cityscapes/Images/leftImg8bit")
    p.add_argument("--gt_root", type=str,
                   default="/home/rvl1421/Datasets/Cityscapes/GT/gtFine")
    p.add_argument("--crop_size", type=int, default=1024)
    # 起點權重
    p.add_argument("--sam_checkpoint", type=str,
                   default="checkpoints/sam_vit_h_4b8939.pth",
                   help="原始 SA-1B ViT-H 權重，作為 encoder fine-tune 起點")
    # 優化
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="decoder/fusion 等臨時頭的主 LR")
    p.add_argument("--encoder_lr", type=float, default=5e-5,
                   help="ViT-H encoder 的 LR（低於主 LR，防災難性遺忘）")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--accumulate_steps", type=int, default=8)
    p.add_argument("--max_norm", type=float, default=1.0)
    # loss
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--lovasz_weight", type=float, default=0.0)
    p.add_argument("--lovasz_start_epoch", type=int, default=0)
    p.add_argument("--mfb", action="store_true", default=True)
    # 執行
    p.add_argument("--eval_interval", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="checkpoints/cityscapes_pretrain")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry_run", action="store_true",
                   help="只跑數個 step 驗證可建構、可前後向、不 OOM")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 模型建構（乾淨版 PairSAM）
# ---------------------------------------------------------------------------
def build_model(args):
    # use_vgg_adapter=False → 不啟用 adapter、forward 不讀 clear_image
    # cond=False           → condition embedding 退化為常數
    cfg = {
        "model_type": "vit_h",
        "use_vgg_adapter": False,
        "cond": False,
        "lrh": True,          # 保留 context_fusion_head 作為臨時頭的一部分
        "decoder": "unified",
        "ref": False,
        "mfb": args.mfb,
    }
    # checkpoint=SA-1B：build 的 state_dict 過濾只會匹配 image_encoder.*（原始 SAM 的
    # prompt/mask decoder 與客製 PairSAM 不同名，會被濾除），故 encoder 載入 SA-1B、
    # decoder/fusion 隨機初始化 —— 正是我們要的 pretrain 起點。
    model = build_pair_sam_from_config(cfg, checkpoint=args.sam_checkpoint)
    model = model.to(args.device)

    # gradient checkpointing（僅影響 self.training 時的 encoder forward）
    model.image_encoder.use_checkpoint = True

    # ── 可訓練範圍：encoder 全量 + 臨時頭；凍結 CLIP backbone 與 CMA fusion_module ──
    for param in model.parameters():
        param.requires_grad = True
    # CLIP 文字塔凍結（只有 text_encoder.projection 保持可訓練）
    for name, param in model.text_encoder.named_parameters():
        param.requires_grad = name.startswith("projection")
    # CMAAlignment（VGG + UAWarpC）本階段完全未使用，凍結避免誤更新
    for param in model.fusion_module.parameters():
        param.requires_grad = False
    # vgg_injector 未啟用（無 hook），凍結以免進入 optimizer
    for param in model.vgg_injector.parameters():
        param.requires_grad = False

    return model


def build_optimizer(model, args):
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": encoder_params, "lr": args.encoder_lr, "name": "encoder"},
        {"params": head_params, "lr": args.lr, "name": "head"},
    ]
    n_enc = sum(p.numel() for p in encoder_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"✅ 可訓練參數: encoder {n_enc/1e6:.1f}M (LR={args.encoder_lr:.1e}) | "
          f"head {n_head/1e6:.1f}M (LR={args.lr:.1e})")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    def lr_lambda(epoch_idx):
        if epoch_idx < args.warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, args.warmup_epochs))
        progress = float(epoch_idx - args.warmup_epochs) / float(
            max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# 前向組裝：把 model 輸出的 per-class low-res logits 組成 19 類高解析度 logits
# ---------------------------------------------------------------------------
def forward_assemble(model, batched_input, out_size):
    """回傳 list，每筆為 (low_res_logits(K,256,256), fused_hr(1,19,out_h,out_w))。"""
    outputs = model(batched_input)
    results = []
    for out in outputs:
        low = out["low_res_logits"].squeeze(0)          # (K, 256, 256)
        full = low.unsqueeze(0)                          # (1, 19, 256, 256)
        fused = model.context_fusion_head(full) if model.use_lrh else full
        fused_hr = F.interpolate(fused, size=out_size, mode="bilinear",
                                 align_corners=False)    # (1, 19, H, W)
        results.append((low, fused_hr))
    return results


def make_record(image_tensor, device):
    """單張影像 → model.forward 期望的 input dict（adapter/cond 皆停用，故欄位最小）。"""
    h, w = image_tensor.shape[-2:]
    return {
        "image": image_tensor.to(device),
        "text_prompts": list(CITYSCAPES_CLASSES),
        "condition_id": torch.tensor(0, device=device),  # cond=False 會忽略其值
        "original_size": (h, w),
    }


# ---------------------------------------------------------------------------
# 訓練
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, mask_loss_fn, context_loss_fn,
                    cls_w, epoch, args):
    model.train()
    context_loss_fn.lovasz_weight = lovasz_weight_for_epoch(
        epoch, args.lovasz_start_epoch, args.lovasz_weight)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    optimizer.zero_grad()
    running = 0.0
    n_batches = len(loader)

    for step, batch in enumerate(pbar):
        images = batch["image"]          # (B, 3, crop, crop)
        labels = batch["label"].to(args.device)   # (B, crop, crop)
        B = images.shape[0]
        batched_input = [make_record(images[i], args.device) for i in range(B)]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            assembled = forward_assemble(model, batched_input, out_size=labels.shape[-2:])

            total_loss = torch.zeros((), device=args.device)
            for i in range(B):
                low, fused_hr = assembled[i]
                gt = labels[i].unsqueeze(0)                    # (1, H, W)
                valid = (gt != IGNORE_INDEX).float().unsqueeze(1)  # (1,1,H,W)

                # ── Dice：每類別 1 個 mask，稀有類以 MFB 權重加權平均 ──
                low_hr = F.interpolate(low.unsqueeze(1), size=labels.shape[-2:],
                                       mode="bilinear", align_corners=False)  # (K,1,H,W)
                K = low_hr.shape[0]
                targets = torch.stack([(gt == c).float() for c in range(K)], dim=0)  # (K,1,H,W)
                valid_k = valid.expand(K, -1, -1, -1)
                mask_loss, _ = mask_loss_fn(low_hr, targets, valid_k)  # (K,1)
                mask_loss = mask_loss.squeeze(1)                        # (K,)
                dice_term = (mask_loss * cls_w).sum() / cls_w.sum()

                # ── CE（+可選 Lovász）on 組裝後 19 類高解析度 logits ──
                ce_term, _, _, _ = context_loss_fn(fused_hr, gt)

                total_loss = total_loss + dice_term + ce_term
            total_loss = total_loss / float(B)

        (total_loss / float(args.accumulate_steps)).backward()

        is_accum = ((step + 1) % args.accumulate_steps == 0)
        is_last = (step + 1 == n_batches)
        if is_accum or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            optimizer.step()
            optimizer.zero_grad()

        running += float(total_loss.item())
        pbar.set_postfix(loss=f"{running/(step+1):.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if args.dry_run and step >= 2:
            print("✅ dry_run：3 個 train step 完成，未 OOM。")
            return running / (step + 1)

    scheduler.step()
    return running / max(1, n_batches)


# ---------------------------------------------------------------------------
# 驗證（tiling：2048 寬切成兩塊 1024×1024）
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, args):
    model.eval()
    calc = SegMetricsCalculator(classes=CITYSCAPES_CLASSES, ignore_index=IGNORE_INDEX)

    for batch in tqdm(loader, desc="[Val]"):
        image = batch["image"][0]          # (3, H, W)，val batch_size=1
        label = batch["label"][0].numpy()  # (H, W)
        H, W = image.shape[-2:]
        pred_full = torch.zeros((H, W), dtype=torch.long)

        # 沿寬度切成 crop_size 的 tile（Cityscapes 2048 → 兩塊；不足處以最後一塊右對齊覆蓋）
        xs = list(range(0, max(1, W - args.crop_size + 1), args.crop_size))
        if xs[-1] != W - args.crop_size:
            xs.append(W - args.crop_size)
        for x0 in xs:
            tile = image[:, :, x0:x0 + args.crop_size]
            record = make_record(tile, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assembled = forward_assemble(model, [record],
                                             out_size=(H, args.crop_size))
            pred_tile = assembled[0][1].squeeze(0).float().argmax(dim=0).cpu()  # (H, crop)
            pred_full[:, x0:x0 + args.crop_size] = pred_tile

        calc.update(pred_full.numpy(), label, condition=None)

    results = calc.compute()
    return results


# ---------------------------------------------------------------------------
def save_encoder(model, path):
    """只抽 image_encoder.* 權重存檔，供 Stage-2 train.py --checkpoint 直接載入。"""
    enc_state = {k: v.cpu() for k, v in model.state_dict().items()
                 if k.startswith("image_encoder.")}
    torch.save({"model_state_dict": enc_state}, path)
    print(f"💾 已保存 encoder-only 權重（{len(enc_state)} tensors）→ {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 資料
    train_set = CityscapesPretrainDataset(args.images_root, args.gt_root, "train",
                                          crop_size=args.crop_size)
    val_set = CityscapesPretrainDataset(args.images_root, args.gt_root, "val",
                                        crop_size=args.crop_size)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 模型 / 優化器 / loss
    model = build_model(args)
    optimizer, scheduler = build_optimizer(model, args)
    mask_loss_fn = MaskLoss(dice_weight=args.dice_weight)
    # Cityscapes MFB 權重：CE 與 Dice 加權平均皆採用，對齊 Cityscapes 類別分佈
    context_loss_fn = ContextLoss(ce_weight=args.ce_weight, lovasz_weight=args.lovasz_weight,
                                  use_mfb=args.mfb,
                                  class_weights=CITYSCAPES_CLASS_WEIGHTS.clone()).to(args.device)
    cls_w = (CITYSCAPES_CLASS_WEIGHTS if args.mfb
             else torch.ones_like(CITYSCAPES_CLASS_WEIGHTS)).to(args.device)

    best_miou = 0.0
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                   mask_loss_fn, context_loss_fn, cls_w, epoch, args)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        if args.dry_run:
            results = validate(model, val_loader, args) if False else None
            save_encoder(model, os.path.join(args.output_dir, "dry_run_encoder.pth"))
            print("✅ dry_run 完成（訓練 3 step + 存檔）。")
            return

        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            results = validate(model, val_loader, args)
            miou = results["miou"]
            print(f"Epoch {epoch+1}: Cityscapes val mIoU = {miou:.4f}")
            save_encoder(model, os.path.join(args.output_dir, "sam_vit_h_cityscapes_encoder_last.pth"))
            if miou > best_miou:
                best_miou = miou
                save_encoder(model, os.path.join(args.output_dir, "sam_vit_h_cityscapes_encoder_best.pth"))
                print(f"🏆 新的 best mIoU = {best_miou:.4f}")

    print(f"\n✅ Stage-1 完成。best Cityscapes val mIoU = {best_miou:.4f}")
    print(f"   encoder 權重：{args.output_dir}/sam_vit_h_cityscapes_encoder_best.pth")
    print(f"   Stage-2 用法：train.py ... --checkpoint {args.output_dir}/sam_vit_h_cityscapes_encoder_best.pth")


if __name__ == "__main__":
    main()
