import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 強制使用 Agg 後端，不啟動 GUI 視窗
import matplotlib.pyplot as plt
import argparse

# --- 導入你的模組 ---
# 注意：請確認你的 dataloader 檔名是 weather_dataloader.py 還是 dataloader.py
# 如果檔名是 weather_dataloader.py，請用下面這行：
from utils.weather_dataloader import WeatherSegmentationDataset
# from utils.dataloader import WeatherSegmentationDataset # (舊的引用方式)

from weather_trainer import WeatherSAMTrainer
from segment_anything.build_weather_sam import build_weather_sam_vit_h, build_weather_sam_vit_b

def plot_history(history, output_dir):
    """將訓練歷史繪製成圖表 (含 Train 與 Val)"""
    if not history:
        return
        
    df = pd.DataFrame(history)
    plt.figure(figsize=(12, 6))
    
    # 繪製 Training Loss
    plt.plot(df['epoch'], df['train_total'], 'b-o', label='Train Total', linewidth=2)
    plt.plot(df['epoch'], df['val_total'], 'r-s', label='Val Total', linewidth=2)
    
    # 繪製 Dice (比較 Train vs Val)
    if 'train_dice' in df.columns and 'val_dice' in df.columns:
        plt.plot(df['epoch'], df['train_dice'], 'g--', label='Train Dice', alpha=0.6)
        plt.plot(df['epoch'], df['val_dice'], 'm--', label='Val Dice', alpha=0.6)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('WeatherSAM Training Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 儲存圖片
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    print(f"📉 Loss curve updated at: {os.path.join(output_dir, 'loss_curve.png')}")

def main():
    # ================= 參數設定 =================
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # SAM 權重路徑
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_b", "vit_h"], help="Model type")
    
    # ★★★ 修改 1: 改為接收 CSV 路徑，而不是資料夾路徑 ★★★
    parser.add_argument("--train_csv", type=str, default="data/weather_dataset/train.csv", help="Path to train CSV mapping")
    parser.add_argument("--val_csv", type=str, default="data/weather_dataset/val.csv", help="Path to val CSV mapping")
    # parser.add_argument("--data_root", ...) # 這一行刪除
    
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs_weather_sam")
    
    args = parser.parse_args()
    
    # ===========================================

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Start Training WeatherSAM ({args.model_type})...")
    print(f"   Device: {args.device}")
    print(f"   Train CSV: {args.train_csv}")
    print(f"   Val CSV:   {args.val_csv}")

    # 1. 建立 WeatherSAM 模型
    print("🏗️  Building model...")
    if args.model_type == "vit_h":
        model = build_weather_sam_vit_h(checkpoint=args.checkpoint)
    else:
        model = build_weather_sam_vit_b(checkpoint=args.checkpoint)
    
    # 2. 準備 Dataset 與 DataLoader
    print("📂 Preparing data from CSVs...")
    
    # ★★★ 修改 2: 使用 csv_file 參數初始化 Dataset ★★★
    # 注意：這裡不再需要 os.path.join 組合路徑，因為 CSV 裡面已經是完整路徑
    
    train_ds = WeatherSegmentationDataset(csv_file=args.train_csv, mode='train')
    
    # 檢查是否提供了 Val CSV，如果沒有，雖然不建議但可暫時用 train 代替或報錯
    if os.path.exists(args.val_csv):
        val_ds = WeatherSegmentationDataset(csv_file=args.val_csv, mode='val')
    else:
        print(f"⚠️ Warning: Validation CSV not found at {args.val_csv}. Using Train set for validation.")
        val_ds = WeatherSegmentationDataset(csv_file=args.train_csv, mode='val')

    # DataLoader 保持不變，記得用自定義的 collate_fn
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn,
        pin_memory=True # 建議開啟，加速資料傳輸到 GPU
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn,
        pin_memory=True
    )
    
    # 3. 初始化訓練器
    print("⚙️  Initializing Trainer...")
    trainer = WeatherSAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr
    )

    # 4. 訓練迴圈 (保持不變)
    best_val_loss = float('inf')
    history = []

    print(f"🔥 Start training loop for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        # --- Training Step ---
        train_metrics = trainer.train_epoch(epoch)
        
        # --- Validation Step ---
        val_metrics = trainer.validate_epoch(epoch)
        
        # --- Logging ---
        log_entry = {
            "epoch": epoch + 1,
            "train_total": train_metrics["total"],
            "train_dice": train_metrics["dice"],
            "val_total": val_metrics["total"],
            "val_dice": val_metrics["dice"]
        }
        history.append(log_entry)
        
        # 印出當前結果
        print(f"   [Result] Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}")
        
        # 儲存 CSV
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
        
        # 繪圖
        plot_history(history, args.output_dir)
        
        # 儲存最佳模型 (依據 Val Loss)
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            save_path = os.path.join(args.output_dir, "weather_sam_best.pth")
            trainer.save_checkpoint(save_path)
            print(f"   🏆 New best model saved! (Val Loss: {best_val_loss:.4f})")
            
        # 定期儲存 Checkpoint (例如每 5 epoch)
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.output_dir, f"weather_sam_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(save_path)

    print("\n✅ Training completed!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()