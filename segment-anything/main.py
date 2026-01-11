import torch
from torch.utils.data import DataLoader
# 修改 import 路徑以匹配你的檔案
from utils.dataloader import WeatherSegmentationDataset 
from trainer import SAMTrainer
from segment_anything.build_sam import build_sam_vit_h
import os

def main():
    # 設定
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth" 
    DATA_ROOT = "data/weather_dataset" 
    BATCH_SIZE = 2
    EPOCHS = 20
    
    # 確保 Checkpoint 存在
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    print(f"Using device: {DEVICE}")

    # 1. 建構模型
    sam_model = build_sam_vit_h(checkpoint=CHECKPOINT_PATH)
    sam_model.to(DEVICE)
    
    # 2. 準備資料
    train_dataset = WeatherSegmentationDataset(root_dir=DATA_ROOT, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 3. 初始化訓練器 (Trainer 內部會處理凍結邏輯)
    trainer = SAMTrainer(
        model=sam_model,
        train_loader=train_loader,
        val_loader=None, 
        device=DEVICE
    )

    # 4. 開始訓練
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        avg_loss = trainer.train_epoch(epoch)
        print(f"Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            save_path = f"sam_weather_finetuned_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()