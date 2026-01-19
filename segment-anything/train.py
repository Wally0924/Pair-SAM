import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 強制使用 Agg 後端，不啟動 GUI 視窗
import matplotlib.pyplot as plt
import argparse

# 導入自定義模組
# 確保這些檔案都在正確的目錄下
from utils.dataloader import WeatherSegmentationDataset 
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
    
    # 請確保這裡的路徑指向標準 SAM 的權重檔
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_b", "vit_h"], help="Model type")
    
    parser.add_argument("--data_root", type=str, default="data/weather_dataset", help="Dataset root (containing train/val folders)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs_weather_sam")
    
    args = parser.parse_args()
    
    # ===========================================

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Start Training WeatherSAM ({args.model_type})...")
    print(f"   Device: {args.device}")
    print(f"   Data Root: {args.data_root}")

    # 1. 建立 WeatherSAM 模型
    print("🏗️  Building model...")
    if args.model_type == "vit_h":
        # 載入 checkpoint 時會自動過濾掉不匹配的 key，只保留 ViT 和 Decoder
        model = build_weather_sam_vit_h(checkpoint=args.checkpoint)
    else:
        model = build_weather_sam_vit_b(checkpoint=args.checkpoint)
    
    # 2. 準備 Dataset 與 DataLoader
    print("📂 Preparing data...")
    
    # 假設您的資料集結構為:
    # data/weather_dataset/
    #   ├── train/ (bad_weather_images, reference_masks, labels)
    #   └── val/   (bad_weather_images, reference_masks, labels)
    
    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val")
    
    # 如果沒有分 train/val 資料夾，暫時都指向同一個 (僅供測試)
    if not os.path.exists(val_root):
        print("⚠️ Warning: 'val' folder not found. Using 'train' for validation (Not recommended).")
        val_root = train_root

    train_ds = WeatherSegmentationDataset(root_dir=train_root, mode='train')
    val_ds = WeatherSegmentationDataset(root_dir=val_root, mode='val')

    # ★★★ 關鍵：必須使用自定義的 collate_fn ★★★
    # 因為 batch 裡面的 'text_prompts' 是 List[List[str]]，預設 collate 會報錯
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn 
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, # 驗證時 Batch 可以大一點，如果不算梯度的話
        shuffle=False, 
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn
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

    # 4. 訓練迴圈
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