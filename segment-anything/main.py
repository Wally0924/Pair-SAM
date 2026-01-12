import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 強制使用 Agg 後端，不啟動 GUI 視窗
import matplotlib.pyplot as plt

# 導入你的自定義模組
from utils.dataloader import WeatherSegmentationDataset 
from trainer import SAMTrainer
from segment_anything.build_sam import build_sam_vit_h

def plot_history(history, output_dir):
    """將訓練歷史繪製成圖表"""
    if not history:
        return
        
    df = pd.DataFrame(history)
    plt.figure(figsize=(12, 6))
    
    # 繪製總 Loss
    plt.plot(df['epoch'], df['total'], 'b-o', label='Total Loss', linewidth=2)
    
    # 繪製細項 Loss (Dice 與 IoU MSE 通常範圍較小，適合一起看)
    plt.plot(df['epoch'], df['dice'], 'r--', label='Dice Loss')
    plt.plot(df['epoch'], df['iou_mse'], 'g--', label='IoU MSE')
    
    # BCE 通常有加權 (例如 x20)，為了方便觀察，繪製其原始值
    plt.plot(df['epoch'], df['bce'], 'y:', label='BCE (unweighted)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('SAM Fine-tuning Loss History (Adverse Weather)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 儲存圖片
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    print(f"Loss curve updated at: {os.path.join(output_dir, 'loss_curve.png')}")

def main():
    # ================= 參數設定 =================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth" 
    DATA_ROOT = "data/weather_dataset/train"   # 確保這是你的訓練集路徑
    BATCH_SIZE = 4                             # 若顯存不足可調至 2
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    OUTPUT_DIR = "outputs"
    # ===========================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 確保權重檔案存在
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: SAM Checkpoint not found at {CHECKPOINT_PATH}")
        return

    print(f"Using device: {DEVICE}")

    # 1. 載入 SAM 模型 (ViT-H 版本)
    print("Loading model...")
    sam_model = build_sam_vit_h(checkpoint=CHECKPOINT_PATH)
    sam_model.to(DEVICE)
    
    # 2. 準備 Dataset 與 DataLoader
    print("Preparing dataset...")
    # 使用測試模式 (max_images=1200) 以加快實驗循環
    train_dataset = WeatherSegmentationDataset(
        root_dir=DATA_ROOT, 
        mode='train', 
        max_images=1200
    ) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # 3. 初始化訓練器
    trainer = SAMTrainer(
        model=sam_model,
        train_loader=train_loader,
        val_loader=None, 
        device=DEVICE
    )

    # 4. 訓練迴圈
    best_loss = float('inf')
    history = []

    print(f"Start training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 執行一個 epoch 的訓練，取得平均指標
        avg_metrics = trainer.train_epoch(epoch)
        
        # 紀錄 Epoch 編號
        avg_metrics['epoch'] = epoch + 1
        history.append(avg_metrics)
        
        # 儲存 CSV 日誌備份
        log_df = pd.DataFrame(history)
        log_df.to_csv(os.path.join(OUTPUT_DIR, "train_log.csv"), index=False)
        
        # 更新 Loss 曲線圖
        plot_history(history, OUTPUT_DIR)
        
        # 印出狀態
        print(f"Avg Total Loss: {avg_metrics['total']:.4f}")
        print(f"Detail -> BCE: {avg_metrics['bce']:.4f}, Dice: {avg_metrics['dice']:.4f}, IoU MSE: {avg_metrics['iou_mse']:.4f}")
        
        # 儲存最優模型
        if avg_metrics['total'] < best_loss:
            best_loss = avg_metrics['total']
            save_path = os.path.join(OUTPUT_DIR, "sam_weather_best.pth")
            trainer.save_checkpoint(save_path)
            print(f"New best model saved with Loss: {best_loss:.4f}")

    print("\nTraining completed!")
    print(f"Best Loss achieved: {best_loss:.4f}")
    print(f"All outputs saved in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()