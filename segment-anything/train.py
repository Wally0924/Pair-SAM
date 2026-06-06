# train.py
import os
import json
# ViT-H global attention (64×64 token grid, 16 heads) 需要 ~1 GiB 連續記憶體，
# expandable_segments 讓 CUDA allocator 以延伸段取代重新分配，大幅降低碎片化 OOM 機率
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# 引入你的模組
from utils.weather_dataloader import WeatherSegmentationDataset
from utils.rare_class_sampler import RareClassSampler
from weather_trainer import WeatherSAMTrainer
from segment_anything.build_weather_sam import build_weather_sam_from_config


def set_seed(seed: int, deterministic: bool = True):
    """
    固定全域隨機源，確保 ablation 可重現。
    deterministic=True 時關閉 cudnn.benchmark（犧牲 5~15% 速度）換取完全可重現的結果。
    探索性實驗可傳 False 保留速度，但不能用於論文 ablation。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker 各自的 numpy / random 也綁定到主 seed 衍生值。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# 📊 1. 繪製圖表功能
# ==========================================
def plot_history(history, output_dir):
    if not history:
        return
    df = pd.DataFrame(history)
    epochs = df['epoch'].tolist()

    # 每個 loss 各自一個子圖，各有自己的 y 軸尺度
    # train_col = f'train_{key}'，val_col = f'val_{key}'；缺失欄位自動跳過
    components = [
        ('total',           'Total Loss',                     'k',         'Train Total',      'Val Total'),
        ('ce',              'CE Loss (unweighted)',           'b',         'Train CE',         'Val CE'),
        ('ce_weighted',     'CE Loss (MFB-weighted)',         'royalblue', 'Train CE_w',       'Val CE_w'),
        ('lovasz',          'Lovász-Softmax Loss ↓',          'darkorange','Train Lovász',     'Val Lovász'),
        ('dice',            'Dice Loss (unweighted)',         'g',         'Train Dice',       'Val Dice'),
        ('dice_weighted',   'Dice Loss (MFB-weighted)',       'seagreen',  'Train Dice_w',     'Val Dice_w'),
        # ── 需求 1: mIoU（僅 val，train 欄位不存在會自動跳過）──
        ('miou',            'Val mIoU ↑ [0→1]',              'darkgreen', 'Train mIoU',       'Val mIoU'),
        ('conf_mean',       'UAWarpC Confidence Mean [0-1]',  'purple',    'Train Conf',       'Val Conf'),
        ('valid_ratio',     'Warp Valid Ratio [0-1]',         'm',         'Train Valid',      'Val Valid'),
        ('pred_conf_mean',  'Prediction Confidence ↑',         'navy',      'Train Pred Conf',  'Val Pred Conf'),
        ('pred_entropy',    'Prediction Entropy ↓ decisive',   'crimson',   'Train Pred Ent',   'Val Pred Ent'),
        ('logit_margin',    'Top1-Top2 Probability Margin ↑',  'sienna',    'Train Margin',     'Val Margin'),
        # ── 需求 2: Per-group gradient norm（僅 train）──
        ('grad_norm_main',    'Grad Norm — main group',       'coral',     'Train GN main',    'Val GN main'),
        ('grad_norm_decoder', 'Grad Norm — decoder head',     'peru',      'Train GN decoder', 'Val GN decoder'),
        # ── 需求 3: AMP scaler scale factor（僅 train；下降 → gradient underflow 警告）──
        ('scaler_scale',    'AMP Scaler Scale ↑ stable',      'gray',      'Train Scale',      'Val Scale'),
        # [testv14] WarpedVGG Adapter 診斷
        ('inject_cos_sim',  'WarpedVGG Inject CosSim ↑ stable','purple',   'Train Inj CosSim', 'Val Inj CosSim'),
        ('inject_gate',     'WarpedVGG softplus(gate) ↑ learning','brown', 'Train Gate',       'Val Gate'),
    ]

    n_plots = len(components)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for ax, (key, title, color, train_label, val_label) in zip(axes, components):
        train_col = f'train_{key}'
        val_col   = f'val_{key}'
        has_data  = False

        if train_col in df.columns:
            ax.plot(epochs, df[train_col], color=color, linestyle='-',
                    linewidth=2, label=train_label, marker='o', markersize=3)
            has_data = True
        if val_col in df.columns:
            ax.plot(epochs, df[val_col], color=color, linestyle='--',
                    linewidth=2, label=val_label, marker='s', markersize=3, alpha=0.8)
            has_data = True

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

        # y 軸下限不低於 0，讓坐標軸更直觀
        if has_data:
            ymin = ax.get_ylim()[0]
            ax.set_ylim(bottom=max(0, ymin * 0.95))

    # 隱藏多餘的空白 subplot
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('WeatherSAM Training Curves (Per-Component)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"📉 Curves updated at: {save_path}")


# ==========================================
# 🖨️ 2. 美觀的參數列印函式
# ==========================================
def print_training_config(args, device):
    print("\n" + "="*60)
    print(f"🚀  WeatherSAM Training Configuration")
    print("="*60)
    
    # 1. 系統與環境
    print(f"🖥️  System Info:")
    print(f"   • Device:            {device}")
    print(f"   • Model Type:        {args.model_type}")
    print(f"   • Checkpoint:        {args.checkpoint if args.checkpoint else 'None (Scratch)'}")
    print(f"   • Output Dir:        {args.output_dir}")
    
    # 2. 訓練超參數
    print(f"\n⚙️  Hyperparameters:")
    print(f"   • Epochs:            {args.epochs}")
    print(f"   • Batch Size:        {args.batch_size}")
    print(f"   • Learning Rate:     {args.lr}")
    print(f"   • Max Norm (Clip):   {args.max_norm}")
    print(f"   • Early Stopping:    Patience {args.patience}, Min Delta {args.min_delta}")
    
    # 3. Decoupled Loss Weights
    print(f"\n⚖️  Decoupled Loss Weights:")
    print(f"   • CE Weight:         {args.ce_weight}")
    print(f"   • Dice Weight:       {args.dice_weight}")
    lovasz_w = getattr(args, 'lovasz_weight', 0.0)
    print(f"   • Lovász Weight:     {lovasz_w}{'  (disabled)' if lovasz_w == 0.0 else '  ← mIoU-aligned'}")
    # print(f"   • IoU Weight:        {args.iou_weight}")  # [Mask2Former] IoU MSE Loss 已移除
    
    # 4. Decoder LR
    print(f"\n🔓  MaskDecoder LR Groups:")
    print(f"   • Decoder LR Scale:      {args.decoder_lr_scale}  (mask head LR       = {args.lr * args.decoder_lr_scale:.2e})")
    print(f"   • Transformer LR Scale:  {args.transformer_lr_scale}  (transformer LR    = {args.lr * args.transformer_lr_scale:.2e})")
    
    # 4. 路徑資訊
    print(f"\n📂  Paths:")
    print(f"   • Train CSV:         {args.train_csv}")
    print(f"   • Val CSV:           {args.val_csv}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser()

    # --- 基礎設定 ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_b", "vit_h"])
    parser.add_argument("--checkpoint", type=str,
                        default="/home/rvl1421/SAM_research-1/segment-anything/checkpoints/sam_vit_h_4b8939.pth",
                        help="Path to checkpoint.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a training checkpoint (.pth) to resume from. If set, --checkpoint is ignored.")
    parser.add_argument("--output_dir", type=str, default="outputs_weather_sam_mask2former_testv15",)
    
    # --- 訓練超參數 ---
    parser.add_argument("--epochs", type=int, default=80, help="總共訓練的 Epoch 數量")
    parser.add_argument("--patience", type=int, default=10, help="提早停止 (Early stopping) 的耐心值")
    parser.add_argument("--min_delta", type=float, default=0.0005,
                        help="標記為顯著 mIoU 進步的門檻；任何 mIoU 新高都會重置 early stopping。")
    parser.add_argument("--batch_size", type=int, default=1, help="每次前向傳播的 Batch size（ViT-H global attention 需大量 VRAM，建議 1）")
    parser.add_argument("--accumulate_steps", type=int, default=4, help="梯度累積步數 (等效 batch_size = batch_size * steps，預設 1×8=8)")
    parser.add_argument("--lr", type=float, default=5e-5, help="學習率")
    parser.add_argument("--max_norm", type=float, default=1.0, help="梯度裁剪的 Max norm。")

    # --- Decoupled Loss 權重 ---
    parser.add_argument("--ce_weight", type=float, default=2.0, help="ContextLoss (CrossEntropy) 權重")
    parser.add_argument("--dice_weight",  type=float, default=1.0,
                        help="MaskLoss (Dice) 權重")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="CE label smoothing（建議 0.05）")
    parser.add_argument("--lovasz_weight", type=float, default=1.0,
                        help="Lovász-Softmax Loss 權重（0.0=停用，退化為純 CE；建議從 0.5 開始實驗）。"
                             "Lovász 直接優化 mIoU 的可微近似，梯度方向與評估指標完全對齊。")
    parser.add_argument("--lovasz_start_epoch", type=int, default=0,
                        help="Lovász 延後啟動：前 N 個 epoch（0-based）停用 Lovász，僅以 CE 預訓，"
                             "第 N 個 epoch（顯示為第 N+1 epoch）起加入 Lovász 微調 mIoU。"
                             "0=從頭啟用（預設，向後相容）。對應 Berman et al. 2018 的 CE 預訓→Lovász fine-tune 協定。")
    parser.add_argument("--warmup_gate_epochs", type=int, default=3,
                        help="前 N epoch 凍結 vgg_injector.gates，讓 main decoder 先穩定")
    # parser.add_argument("--iou_weight", type=float, default=3.0, help="IoU MSE Loss 權重")  # [Mask2Former] 已移除

    # --- Dense Gate LR ---
    # [testv14] gate_lr_scale 已移除（dense_gate 已隨 dense_ref 路徑下架）

    # --- LR Warmup ---
    parser.add_argument("--warmup_epochs", type=int, default=5, help="LR Warmup 線性上升的 Epoch 數")

    # --- 可重現性 ---
    parser.add_argument("--seed", type=int, default=42, help="全域隨機種子（確保 ablation 可重現）")
    parser.add_argument("--no_deterministic", action="store_true",
                        help="關閉 cudnn deterministic 以換取訓練速度（探索性實驗用；論文 ablation 勿用）")

    # --- Decoder LR ---
    parser.add_argument("--decoder_lr_scale", type=float, default=0.5,
                        help="MaskDecoder mask_tokens 相對主幹 LR 的縮放比例 (預設 0.1 = 1/10 LR)")
    parser.add_argument("--transformer_lr_scale", type=float, default=0.05,
                        help="MaskDecoder Transformer weights 相對主幹 LR 的縮放比例 (0.05 = 1/20 LR，讓 transformer 適應 weather-fused feature 分布)")

    # --- Image Encoder 部分解凍 ---
    parser.add_argument("--unfreeze_encoder_blocks", type=int, default=0,
                        help="解凍 ViT-H Image Encoder 最後 N 個 Block（預設 0=全凍結）。"
                             "建議從 2 開始：解凍 Block 30-31 讓語意嵌入層適應天氣域，"
                             "Block 0-29 繼續凍結以保留 SAM 通用分割能力。")
    parser.add_argument("--encoder_lr_scale", type=float, default=0.005,
                        help="解凍的 Image Encoder Block 相對主幹 LR 的縮放比例（預設 0.005 = 1/200 LR）。"
                             "極低 LR 確保語意嵌入緩慢調適，不破壞預訓練通用特徵。")

    # --- 資料路徑 ---
    parser.add_argument("--train_csv", type=str, default="/home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_train.csv",
                        help="訓練資料集 CSV 路徑")
    parser.add_argument("--val_csv", type=str, default="/home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_val.csv",
                        help="驗證資料集 CSV 路徑")

    # [testv14] WarpedVGG Adapter 注入 SAM ViT-H Encoder（預設啟用，--no-use_vgg_adapter 關閉）
    parser.add_argument("--use_vgg_adapter", action=argparse.BooleanOptionalAction, default=True,
                        help="啟用 MultiStage WarpedVGG Adapter（在 ViT-H Block 7/15/23/31 各注入對齊 VGG 特徵）")
    parser.add_argument("--adapter_lr_scale", type=float, default=3.0,
                        help="WarpedVGG Adapter 參數的 LR 倍率（相對 main lr）")

    # --- [ablation] 消融開關（預設值 = FULL，向後相容）---
    parser.add_argument("--inject", choices=["pre", "post"], default="pre",
                        help="WarpedVGG Adapter 注入位置：pre=block 自注意力前；post=後")
    parser.add_argument("--decoder", choices=["unified", "per_class"], default="unified",
                        help="解碼模式：unified=統一查詢；per_class=逐類別獨立解碼")
    parser.add_argument("--lrh", action=argparse.BooleanOptionalAction, default=True,
                        help="是否套用 LRH (ResidualDWConvFusion)；--no-lrh 關閉")
    parser.add_argument("--mfb", action=argparse.BooleanOptionalAction, default=True,
                        help="CE/dice 是否套用 MFB 類別權重；--no-mfb = uniform")
    parser.add_argument("--ref", action=argparse.BooleanOptionalAction, default=True,
                        help="Adapter 是否引入 reference K/V；--no-ref = 零張量")
    parser.add_argument("--rcs", action=argparse.BooleanOptionalAction, default=False,
                        help="Rare Class Sampling：依稀有類過取樣訓練影像（預設關閉；對照實驗顯示 MFB-only 較佳，"
                             "RCS 已從 FULL 移除，見 docs/experiments/2026-06-06-mfb-vs-rcs-comparison.md）")
    parser.add_argument("--rcs_temp", type=float, default=0.01,
                        help="RCS 溫度 T（DAFormer 預設 0.01）")
    parser.add_argument("--class_presence", type=str, default=None,
                        help="class_presence.json 路徑（預設取 train_csv 同目錄）")

    # NOTE: --use_condition_embedding 已移除；模型現在固定使用 condition_encoder（ACDC 模式）
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ★ 必須在建立模型 / DataLoader 之前固定 seed，才能涵蓋權重初始化與 shuffle 順序
    set_seed(args.seed, deterministic=not args.no_deterministic)
    print(f"🎲 Seed fixed: {args.seed} (deterministic={not args.no_deterministic})")

    print_training_config(args, args.device)

    # 1. 建立模型（依消融 config 統一建構；若有 resume 則從訓練 checkpoint 載入權重）
    model_checkpoint = args.resume if args.resume else args.checkpoint
    print(f"🏗️  Building model from: {model_checkpoint}")
    abl_cfg = dict(
        model_type=args.model_type,
        use_vgg_adapter=args.use_vgg_adapter,
        inject=args.inject,
        decoder=args.decoder,
        lrh=args.lrh,
        mfb=args.mfb,
        ref=args.ref,
    )
    model = build_weather_sam_from_config(abl_cfg, checkpoint=model_checkpoint)
    print(f"[Ablation] config = {abl_cfg}")
    if args.use_vgg_adapter:
        print(f"[Config] WarpedVGG Adapter enabled (inject={args.inject}, "
              f"lr_scale={args.adapter_lr_scale})")

    # 落地完整消融 config（含 seed 與 loss 權重），供 eval 重建模型與審計
    with open(os.path.join(args.output_dir, "ablation_config.json"), "w") as f:
        json.dump({**abl_cfg, "seed": args.seed,
                   "lovasz_weight": args.lovasz_weight,
                   "lovasz_start_epoch": args.lovasz_start_epoch,
                   "dice_weight": args.dice_weight,
                   "rcs": args.rcs,
                   "rcs_temp": args.rcs_temp,
                   "epochs": args.epochs,
                   "lr": args.lr,
                   "warmup_epochs": args.warmup_epochs}, f, indent=2)

    # 2. 準備 DataLoader
    print("📂 Preparing data...")

    # 當 Image Encoder 解凍訓練時，必須強制使用原始影像（不能 bypass encoder 走 cache）
    # 否則 encoder blocks 永遠不被呼叫，梯度無法回傳，解凍形同虛設
    # [testv14] WarpedVGG Adapter 啟用時同樣需要 force_raw：hook 註冊在 Block 31 上，
    # 若走 cache 模式 image_encoder 整個被略過，hook 永遠不會觸發，adapter 形同未啟用
    force_raw = (
        getattr(args, 'unfreeze_encoder_blocks', 0) > 0
        or getattr(args, 'use_vgg_adapter', False)
    )
    if force_raw:
        reason = []
        if getattr(args, 'unfreeze_encoder_blocks', 0) > 0:
            reason.append("unfreeze_encoder_blocks>0")
        if getattr(args, 'use_vgg_adapter', False):
            reason.append("use_vgg_adapter=True")
        print(f"⚠️  force_raw_images=True ({' / '.join(reason)}) — cache bypassed for both train & val")

    train_ds = WeatherSegmentationDataset(
        csv_file=args.train_csv,
        image_size=1024,
        mode='train',
        force_raw_images=force_raw,
    )

    val_ds = WeatherSegmentationDataset(
        csv_file=args.val_csv,
        image_size=1024,
        mode='val',
        force_raw_images=force_raw,
    )

    # ★ DataLoader generator 綁定 seed，確保每個 epoch 的 shuffle 順序可重現
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_sampler = None
    if args.rcs:
        cp_path = args.class_presence or os.path.join(
            os.path.dirname(args.train_csv), "class_presence.json")
        if not os.path.isfile(cp_path):
            raise FileNotFoundError(
                f"--rcs 啟用但找不到 {cp_path}；請先執行 "
                f"scripts/precompute_class_presence.py（見 ABLATION_RUNBOOK Phase 0）。")
        with open(cp_path) as f:
            cp = json.load(f)
        num_classes = int(cp.get('num_classes', 19))
        gt_paths = train_ds.data['gt_path'].tolist()
        presence_list = []
        for gp in gt_paths:
            if gp not in cp['presence']:
                raise KeyError(f"class_presence.json 缺少 {gp}；請以 --force 重新 precompute。")
            presence_list.append(cp['presence'][gp])
        # JSON 載入後 class_pixel_counts 的 key 為字串
        pcounts = cp['class_pixel_counts']
        pixel_counts = [int(pcounts.get(str(c), pcounts.get(c, 0))) for c in range(num_classes)]
        train_sampler = RareClassSampler(
            presence_list, pixel_counts, num_samples=len(train_ds),
            temperature=args.rcs_temp, seed=args.seed, num_classes=num_classes)
        top5 = sorted(range(num_classes), key=lambda c: -train_sampler.class_probs[c])[:5]
        print(f"[RCS] enabled (T={args.rcs_temp}); top-5 sampled classes = {top5}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )
    
    # 3. 初始化 Trainer
    print("⚙️  Initializing Trainer...")
    trainer = WeatherSAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )

    # Cityscapes 19 類縮寫（per-class IoU 列印用）
    _CLS_NAMES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'light', 'sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
    ]

    # 4. 訓練迴圈
    start_epoch = 0
    best_val_miou = 0.0   # 以 mIoU 為準（越高越好），取代舊的 loss-based 判斷
    early_stop_counter = 0
    history = []

    # --- Resume: 從中斷點恢復訓練 ---
    if args.resume:
        print(f"🔄 Resuming training from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)  # False: optimizer state uses pickle
        trainer.optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        start_epoch = resume_ckpt.get('epoch', 0)
        best_val_miou = resume_ckpt.get('best_score', 0.0)
        print(f"   ✅ Resumed from epoch {start_epoch}, best_val_miou={best_val_miou*100:.2f}%")

    print(f"🔥 Start training loop for epochs {start_epoch + 1} ~ {args.epochs}")

    is_best = False  # guard against UnboundLocalError if first epoch doesn't trigger save
    for epoch in range(start_epoch, args.epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate_epoch(epoch)
        
        log_entry = {
            "epoch": epoch + 1,
            "train_total":           train_metrics["total"],
            "train_ce":              train_metrics["ce"],
            "train_ce_weighted":     train_metrics.get("ce_weighted", 0.0),
            "train_lovasz":          train_metrics.get("lovasz", 0.0),
            "train_dice":            train_metrics["dice"],
            "train_dice_weighted":   train_metrics.get("dice_weighted", 0.0),
            "val_total":             val_metrics["total"],
            "val_ce":                val_metrics["ce"],
            "val_ce_weighted":       val_metrics.get("ce_weighted", 0.0),
            "val_lovasz":            val_metrics.get("lovasz", 0.0),
            "val_dice":              val_metrics["dice"],
            "val_dice_weighted":     val_metrics.get("dice_weighted", 0.0),
            # ── 需求 1 ──
            "val_miou":              val_metrics.get("miou", 0.0),
            # ── 需求 2 ──
            "train_grad_norm_main":    train_metrics.get("grad_norm_main",    0.0),
            "train_grad_norm_decoder": train_metrics.get("grad_norm_decoder", 0.0),
            # ── 需求 3 ──
            "train_scaler_scale":    train_metrics.get("scaler_scale", 65536.0),
            # CMA 對齊診斷
            "train_conf_mean":       train_metrics.get("conf_mean",    0.0),
            "val_conf_mean":         val_metrics.get("conf_mean",      0.0),
            "train_valid_ratio":     train_metrics.get("valid_ratio",  0.0),
            "val_valid_ratio":       val_metrics.get("valid_ratio",    0.0),
            # dense prediction 診斷：破碎/猶豫通常會反映在 entropy 高、margin 低
            "train_pred_conf_mean":  train_metrics.get("pred_conf_mean", 0.0),
            "val_pred_conf_mean":    val_metrics.get("pred_conf_mean",   0.0),
            "train_pred_entropy":    train_metrics.get("pred_entropy",   0.0),
            "val_pred_entropy":      val_metrics.get("pred_entropy",     0.0),
            "train_logit_margin":    train_metrics.get("logit_margin",   0.0),
            "val_logit_margin":      val_metrics.get("logit_margin",     0.0),
            # [testv14] WarpedVGG Adapter 診斷
            "train_inject_cos_sim":  train_metrics.get("inject_cos_sim", 1.0),
            "val_inject_cos_sim":    val_metrics.get("inject_cos_sim",   1.0),
            "train_inject_gate":     train_metrics.get("inject_gate",    0.0),
            "val_inject_gate":       val_metrics.get("inject_gate",      0.0),
            # Adapter 深度診斷
            "train_inject_delta_norm": train_metrics.get("inject_delta_norm", 0.0),
            "val_inject_delta_norm":   val_metrics.get("inject_delta_norm",   0.0),
            "train_head_delta_norm":   train_metrics.get("head_delta_norm",   0.0),
            "val_head_delta_norm":     val_metrics.get("head_delta_norm",     0.0),
            # per-stage gate（s0–s3）
            **{f"train_inject_gate_s{i}": train_metrics.get(f"inject_gate_s{i}", 0.0) for i in range(4)},
            **{f"val_inject_gate_s{i}":   val_metrics.get(  f"inject_gate_s{i}", 0.0) for i in range(4)},
            # per-stage cos_sim（s0–s3）
            **{f"train_inject_cos_s{i}": train_metrics.get(f"inject_cos_s{i}", 1.0) for i in range(4)},
            **{f"val_inject_cos_s{i}":   val_metrics.get(  f"inject_cos_s{i}", 1.0) for i in range(4)},
        }
        per_cls_iou = val_metrics.get("per_class_iou", [])
        for cls_name, cls_iou in zip(_CLS_NAMES, per_cls_iou):
            log_entry[f"val_iou_{cls_name}"] = cls_iou
        history.append(log_entry)

        val_miou    = val_metrics.get('miou', 0.0)
        conf_mean   = train_metrics.get('conf_mean', 0.0)
        valid_ratio = train_metrics.get('valid_ratio', 0.0)
        gn_m        = train_metrics.get('grad_norm_main', 0.0)
        gn_d        = train_metrics.get('grad_norm_decoder', 0.0)
        scale       = train_metrics.get('scaler_scale', 0.0)
        pred_ent    = val_metrics.get('pred_entropy', 0.0)
        pred_margin = val_metrics.get('logit_margin', 0.0)
        pred_conf   = val_metrics.get('pred_conf_mean', 0.0)
        inj_cos     = val_metrics.get('inject_cos_sim', 1.0)
        inj_gate    = val_metrics.get('inject_gate', 0.0)
        lov_val = val_metrics.get('lovasz', 0.0)
        lov_str = f", Lovász:{lov_val:.4f}" if lov_val > 0.0 else ""
        print(f"   [Epoch {epoch+1}] Train Total: {train_metrics['total']:.4f} | Val Total: {val_metrics['total']:.4f}")
        print(f"               (CE:{val_metrics['ce']:.4f}{lov_str}, Dice:{val_metrics['dice']:.4f})")
        print(f"               [Val mIoU] {val_miou*100:.2f}%")
        # Per-class IoU（縮寫格式，每類別顯示至小數點後一位）
        per_cls = val_metrics.get('per_class_iou', [])
        if per_cls:
            short = [f"{n[:5]}:{v*100:.1f}" for n, v in zip(_CLS_NAMES, per_cls)]
            print(f"               [per-cls] {' '.join(short)}")
        print(f"               [CMA] conf={conf_mean:.3f} | valid={valid_ratio:.3f}")
        print(f"               [pred] conf={pred_conf:.3f} | entropy={pred_ent:.3f} | margin={pred_margin:.3f}")
        print(f"               [Adapter] inject_cos={inj_cos:.4f} gate={inj_gate:.4f}")
        print(f"               [grad] GN_main={gn_m:.3f}  GN_decoder={gn_d:.3f} | AMP scale={scale:.0f}")
        
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
        plot_history(history, args.output_dir)
        
        # ── 需求 1: 以 Val mIoU 作為 best model 與 early stopping 判斷基準（越高越好）──
        current_val_miou = val_metrics.get('miou', 0.0)

        miou_improvement = current_val_miou - best_val_miou

        if miou_improvement > 0:
            # 只要 mIoU 創新高就重置 patience；避免穩定小幅進步被 min_delta 誤殺。
            is_significant = miou_improvement >= args.min_delta
            best_val_miou = current_val_miou
            early_stop_counter = 0
            is_best = True
            if is_significant:
                print(f"   ✅ mIoU improved by {miou_improvement*100:.3f} pp (>= min_delta). Patience reset.")
            else:
                print(f"   ✅ mIoU improved by {miou_improvement*100:.3f} pp (< min_delta, still reset patience).")
        else:
            # 沒有進步
            early_stop_counter += 1
            is_best = False
            print(f"   ⚠️ No mIoU progress ({current_val_miou*100:.2f}% vs best {best_val_miou*100:.2f}%). "
                  f"Counter: {early_stop_counter} / {args.patience}")

        # 只要打破最佳紀錄就存檔（只保留單一最佳權重，覆寫固定檔名，不另存 epoch 命名檔）
        if is_best:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            fixed_path = os.path.join(args.output_dir, "weather_sam_best_latest.pth")
            trainer.save_checkpoint(fixed_path, epoch=epoch+1, best_score=best_val_miou)
            print(f"   🏆 New best (E{epoch+1}, mIoU {best_val_miou*100:.2f}%, LR {current_lr:.1e}) "
                  f"saved: weather_sam_best_latest.pth")

        # 觸發 Early Stopping
        if early_stop_counter >= args.patience:
            print(f"\n🛑 Early stopping triggered! Val mIoU hasn't improved by {args.min_delta} for {args.patience} epochs.")
            break

    print("\n✅ Fine-Tuning completed!")

if __name__ == "__main__":
    main()
