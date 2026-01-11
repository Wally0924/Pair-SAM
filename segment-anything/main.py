import torch
from torch.utils.data import DataLoader
from utils.dataloader import WeatherSegmentationDataset 
from trainer import SAMTrainer
from segment_anything.build_sam import build_sam_vit_h # 確保使用正確的版本 (ViT-B)
import os

def main():
    # 設定
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth" 
    DATA_ROOT = "data/weather_dataset/val" 
    BATCH_SIZE = 4
    EPOCHS = 20
    
    # 定義輸出的資料夾
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    # 3. 初始化訓練器
    trainer = SAMTrainer(
        model=sam_model,
        train_loader=train_loader,
        val_loader=None, 
        device=DEVICE
    )

    # ### 修改重點 1: 初始化一個無限大的 Loss，用來記錄目前看過最低的 Loss
    best_loss = float('inf')

    # 4. 開始訓練
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 訓練一個 epoch 並取得平均 loss
        avg_loss = trainer.train_epoch(epoch)
        print(f"Average Loss: {avg_loss:.4f}")
        
        # 如果現在這個 epoch 的 loss 比我們記錄過的 best_loss 還要低
        if avg_loss < best_loss:
            best_loss = avg_loss # 更新最佳紀錄
            
            # 定義存檔路徑 (固定檔名，這樣就會直接覆蓋舊檔案，只留一份)
            save_path = os.path.join(OUTPUT_DIR, "sam_weather_best.pth")
            
            trainer.save_checkpoint(save_path)
            print(f"✅ New best model saved! (Loss: {best_loss:.4f})")
        else:
            print(f"Loss did not improve (Best: {best_loss:.4f})")

if __name__ == "__main__":
    main()