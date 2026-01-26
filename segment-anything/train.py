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

def plot_history(history, output_dir):
    """繪製訓練曲線"""
    if not history:
        return
    df = pd.DataFrame(history)
    plt.figure(figsize=(12, 6))
    
    # 繪製 Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_total'], 'b-o', label='Train Total Loss', linewidth=2)
    plt.plot(df['epoch'], df['val_total'], 'r-s', label='Val Total Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 繪製 Dice Score 曲線 (將 Loss 轉為 Score: 1 - Loss)
    plt.subplot(1, 2, 2)
    if 'train_dice' in df.columns and 'val_dice' in df.columns:
        # 注意: 這裡假設 log 中的 dice 是 Loss，所以用 1 減去它來畫 Score
        train_score = 1.0 - df['train_dice']
        val_score = 1.0 - df['val_dice']
        plt.plot(df['epoch'], train_score, 'g--', label='Train Dice Score', alpha=0.6)
        plt.plot(df['epoch'], val_score, 'm--', label='Val Dice Score', alpha=0.6)
    
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Accuracy Curve (Dice Score)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'))
    plt.close()
    print(f"📉 Curves updated at: {os.path.join(output_dir, 'training_curve.png')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # [修改 1] 預設載入上次最好的模型 (請確認你的路徑是否正確)
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/rvl1421/SAM_research/segment-anything/checkpoints/sam_vit_h_4b8939.pth", 
                        help="Path to checkpoint. Point this to your best trained model.")
    
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_b", "vit_h"])
    
    parser.add_argument("--train_csv", type=str, default="/home/rvl1421/SAM_research/Datasets/train_all_cached.csv", 
                        help="Path to train CSV.")
    parser.add_argument("--val_csv", type=str, default="/home/rvl1421/SAM_research/Datasets/val_all_cached.csv", 
                        help="Path to val CSV.")
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    
    # [修改 2] 微調時降低 Learning Rate (原為 5e-5)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower for fine-tuning)")
    
    # [修改 3] 更改輸出目錄名稱，避免覆蓋舊實驗
    parser.add_argument("--output_dir", type=str, default="outputs_weather_sam_all_data_testv4")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Start Fine-Tuning WeatherSAM ({args.model_type})...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Output Dir: {args.output_dir}")
    
    # 1. 建立模型
    print("🏗️  Building model...")
    if args.model_type == "vit_h":
        model = build_weather_sam_vit_h(checkpoint=args.checkpoint)
    else:
        model = build_weather_sam_vit_b(checkpoint=args.checkpoint)
    
    # 2. 準備 DataLoader
    print("📂 Preparing data...")
    
    if not os.path.exists(args.train_csv) and "_cached" in args.train_csv:
        print(f"⚠️ Cached CSV not found. Falling back to non-cached version...")
        args.train_csv = args.train_csv.replace("_cached.csv", ".csv")
        
    if not os.path.exists(args.val_csv) and "_cached" in args.val_csv:
        args.val_csv = args.val_csv.replace("_cached.csv", ".csv")

    train_ds = WeatherSegmentationDataset(csv_file=args.train_csv, mode='train')
    
    if os.path.exists(args.val_csv):
        val_ds = WeatherSegmentationDataset(csv_file=args.val_csv, mode='val')
    else:
        print(f"⚠️ Warning: Validation CSV not found. Using Train set for validation.")
        val_ds = WeatherSegmentationDataset(csv_file=args.train_csv, mode='val')

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
        device=args.device,
        lr=args.lr
    )

    # 4. 訓練迴圈
    best_val_loss = float('inf')
    history = []

    print(f"🔥 Start training loop for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate_epoch(epoch)
        
        log_entry = {
            "epoch": epoch + 1,
            "train_total": train_metrics["total"],
            "train_dice": train_metrics["dice"],
            "val_total": val_metrics["total"],
            "val_dice": val_metrics["dice"]
        }
        history.append(log_entry)
        
        print(f"   [Epoch {epoch+1}] Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}")
        
        # 存檔與繪圖
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
        plot_history(history, args.output_dir)
        
        # ------------------------------------------------------------------
        # [修改 4] 儲存最佳模型 (包含詳細資訊)
        # ------------------------------------------------------------------
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            
            # 取得當前 LR
            current_lr = trainer.optimizer.param_groups[0]['lr']
            # 計算 Dice Score (假設 metrics 存的是 Loss)
            current_dice_score = 1.0 - val_metrics['dice']
            
            # 建立詳細檔名
            save_filename = f"best_E{epoch+1}_Dice{current_dice_score:.4f}_LR{current_lr:.1e}.pth"
            save_path = os.path.join(args.output_dir, save_filename)
            
            trainer.save_checkpoint(save_path)
            print(f"   🏆 New best model saved: {save_filename}")
            
            # 同時存一份固定名字的檔案，方便之後使用
            fixed_path = os.path.join(args.output_dir, "weather_sam_best_latest.pth")
            trainer.save_checkpoint(fixed_path)
            
        # # 定期備份 (每 5 Epoch)
        # if (epoch + 1) % 5 == 0:
        #     save_path = os.path.join(args.output_dir, f"weather_sam_epoch_{epoch+1}.pth")
        #     trainer.save_checkpoint(save_path)

    print("\n✅ Fine-Tuning completed!")

if __name__ == "__main__":
    main()