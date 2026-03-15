# train.py
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# 引入你的模組
from utils.weather_dataloader import WeatherSegmentationDataset
from weather_trainer import WeatherSAMTrainer
from segment_anything.build_weather_sam import build_weather_sam_vit_h, build_weather_sam_vit_b

# ==========================================
# 📊 1. 繪製圖表功能
# ==========================================
def plot_history(history, output_dir):
    # (省略內容，與原檔相同)
    if not history:
        return
    df = pd.DataFrame(history)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 1, 1)
    if 'train_total' in df.columns and 'val_total' in df.columns:
        plt.plot(df['epoch'], df['train_total'], 'k-', label='Train Total', linewidth=2)
        plt.plot(df['epoch'], df['val_total'], 'k--', label='Val Total', linewidth=2)
    if 'val_ce' in df.columns:
        plt.plot(df['epoch'], df['val_ce'], 'b-s', label='Val CE', alpha=0.7)
    if 'val_focal' in df.columns:
        plt.plot(df['epoch'], df['val_focal'], 'r-s', label='Val Focal', alpha=0.7)
    if 'val_dice' in df.columns:
        plt.plot(df['epoch'], df['val_dice'], 'g-s', label='Val Dice', alpha=0.7)
    if 'val_iou' in df.columns:
        plt.plot(df['epoch'], df['val_iou'], 'm-^', label='Val IoU MSE', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Decoupled Multi-Loss Components')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(save_path)
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
    print(f"   • Focal Weight:      {args.focal_weight}")
    print(f"   • Dice Weight:       {args.dice_weight}")
    print(f"   • IoU Weight:        {args.iou_weight}")
    
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
    parser.add_argument("--output_dir", type=str, default="outputs_weather_sam_all_data_testv12")
    
    # --- 訓練超參數 ---
    parser.add_argument("--epochs", type=int, default=50, help="總共訓練的 Epoch 數量")
    parser.add_argument("--patience", type=int, default=5, help="提早停止 (Early stopping) 的耐心值")
    parser.add_argument("--min_delta", type=float, default=0.01, help="判定為進步的最小 Loss 差異")
    parser.add_argument("--batch_size", type=int, default=2, help="每次前向傳播的 Batch size")
    parser.add_argument("--accumulate_steps", type=int, default=4, help="梯度累積步數 (等效 batch_size = batch_size * steps)")
    parser.add_argument("--lr", type=float, default=5e-5, help="學習率")
    parser.add_argument("--max_norm", type=float, default=2.0, help="梯度裁剪的 Max norm。")
    parser.add_argument("--gps_noise", type=float, default=0.00005, help="訓練時添加到 GPS 座標的常態分布噪音標準差。")
    
    # --- Decoupled Loss 權重 ---
    parser.add_argument("--ce_weight", type=float, default=1.0, help="ContextLoss (CrossEntropy) 權重")
    parser.add_argument("--focal_weight", type=float, default=1.0, help="MaskLoss (Focal) 權重")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="MaskLoss (Dice) 權重")
    parser.add_argument("--iou_weight", type=float, default=1.0, help="IoU MSE Loss 權重")

    # --- 資料路徑 ---
    parser.add_argument("--train_csv", type=str, default="/home/rvl1421/SAM_research-1/Datasets/train_with_gps.csv", 
                        help="訓練資料集 CSV 路徑")
    parser.add_argument("--val_csv", type=str, default="/home/rvl1421/SAM_research-1/Datasets/val_with_gps.csv", 
                        help="驗證資料集 CSV 路徑")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print_training_config(args, args.device)

    # 1. 建立模型
    print("🏗️  Building model...")
    if args.model_type == "vit_h":
        model = build_weather_sam_vit_h(checkpoint=args.checkpoint)
    else:
        model = build_weather_sam_vit_b(checkpoint=args.checkpoint)
    
    # 2. 準備 DataLoader
    print("📂 Preparing data...")

    train_ds = WeatherSegmentationDataset(
        csv_file=args.train_csv,
        image_size=1024,
        mode='train', 
        gps_noise=args.gps_noise
    )
    
    val_ds = WeatherSegmentationDataset(
        csv_file=args.val_csv, 
        image_size=1024, 
        mode='val', 
        gps_noise=args.gps_noise
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. 初始化 Trainer
    print("⚙️  Initializing Trainer...")
    trainer = WeatherSAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )

    # 4. 訓練迴圈
    print(f"🔥 Start training loop for {args.epochs} epochs")

    best_val_loss = float('inf')
    early_stop_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate_epoch(epoch)
        
        log_entry = {
            "epoch": epoch + 1,
            "train_total": train_metrics["total"],
            "train_ce": train_metrics["ce"],
            "train_focal": train_metrics["focal"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_total": val_metrics["total"],
            "val_ce": val_metrics["ce"],
            "val_focal": val_metrics["focal"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"]
        }
        history.append(log_entry)
        
        print(f"   [Epoch {epoch+1}] Train Total: {train_metrics['total']:.4f} | Val Total: {val_metrics['total']:.4f}")
        print(f"               (CE:{val_metrics['ce']:.4f}, Focal:{val_metrics['focal']:.4f}, Dice:{val_metrics['dice']:.4f}, IoU:{val_metrics['iou']:.4f})")
        
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
        plot_history(history, args.output_dir)
        
        # 儲存最佳模型與 Early Stopping 邏輯
        current_val_loss = val_metrics['total']
        
        # 判斷是否進步 (打破最佳紀錄)
        if current_val_loss < (best_val_loss - args.min_delta):
            # 顯著進步，重置耐心值計數器並更新最佳紀錄
            best_val_loss = current_val_loss
            early_stop_counter = 0
            is_best = True
        elif current_val_loss < best_val_loss:
            # 雖然打破紀錄，但進步幅度小於 min_delta，更新紀錄，但不重置計數器
            best_val_loss = current_val_loss
            early_stop_counter += 1
            is_best = True
            print(f"   ⚠️ Progress but < min_delta. Early Stopping Counter: {early_stop_counter} / {args.patience}")
        else:
            # 沒有進步
            early_stop_counter += 1
            is_best = False
            print(f"   ⚠️ No progress. Early Stopping Counter: {early_stop_counter} / {args.patience} (min_delta={args.min_delta})")
            
        # 只要打破最佳紀錄就存檔
        if is_best:
            
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            save_filename = f"best_E{epoch+1}_TotalLoss{best_val_loss:.4f}_LR{current_lr:.1e}.pth"
            save_path = os.path.join(args.output_dir, save_filename)
            
            trainer.save_checkpoint(save_path, epoch=epoch+1, best_score=best_val_loss)
            print(f"   🏆 New best model saved: {save_filename}")
            
            fixed_path = os.path.join(args.output_dir, "weather_sam_best_latest.pth")
            trainer.save_checkpoint(fixed_path, epoch=epoch+1, best_score=best_val_loss)
            
        # 觸發 Early Stopping
        if early_stop_counter >= args.patience:
            print(f"\n🛑 Early stopping triggered! Validation loss hasn't improved by {args.min_delta} for {args.patience} epochs.")
            break

    print("\n✅ Fine-Tuning completed!")

if __name__ == "__main__":
    main()