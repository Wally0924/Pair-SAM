import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import sys
import numpy as np
import random

# --- 導入你的模組 ---
# 為了避免 import 錯誤，請確保目錄結構正確
from segment_anything.build_weather_sam import build_weather_sam_vit_h
from weather_trainer import WeatherSAMTrainer
# 嘗試導入 collate_fn，如果失敗則使用本地定義的 fallback
try:
    from utils.weather_dataloader import WeatherSegmentationDataset
    collate_fn = WeatherSegmentationDataset.collate_fn
    print("✅ Successfully imported real collate_fn from utils.dataloader")
except ImportError:
    print("⚠️ Could not import utils.dataloader. Using local fallback for collate_fn.")
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        ref_masks = torch.stack([item['reference_mask'] for item in batch])
        gt_masks = torch.stack([item['gt_mask'] for item in batch])
        text_prompts = [item['text_prompts'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        return {
            "image": images,
            "reference_mask": ref_masks,
            "gt_mask": gt_masks,
            "text_prompts": text_prompts,
            "original_size": original_sizes
        }

# ==========================================
# 1. 定義 Mock Data (模擬數據生成器)
# ==========================================
class MockWeatherDataset(Dataset):
    """
    生成隨機數據以模擬 WeatherSegmentationDataset 的輸出。
    用於驗證 Pipeline 是否能跑通，而不需讀取真實硬碟檔案。
    """
    def __init__(self, length=4, image_size=1024):
        self.length = length
        self.image_size = image_size
        self.dummy_prompts = [
            ["car", "road"], 
            ["building", "sky", "tree"], 
            ["person", "sidewalk"],
            ["traffic light"]
        ]
        print(f"🛠️  MockDataset initialized with {length} samples. Image Size: {image_size}x{image_size}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 模擬正規化後的影像 (3, 1024, 1024)
        image = torch.randn(3, self.image_size, self.image_size)
        
        # 模擬參考遮罩 (3, 1024, 1024) - 假設是二值化或 RGB Mask
        ref_mask = torch.rand(3, self.image_size, self.image_size)
        
        # 模擬 Ground Truth ID Map (1024, 1024) - 類別 0~18
        gt_mask = torch.randint(low=0, high=19, size=(self.image_size, self.image_size)).long()
        
        # 隨機選取 Prompt
        prompts = self.dummy_prompts[idx % len(self.dummy_prompts)]
        
        return {
            "image": image,
            "reference_mask": ref_mask,
            "gt_mask": gt_mask,
            "text_prompts": prompts,
            "original_size": (self.image_size, self.image_size)
        }

# ==========================================
# 2. 輔助列印函數
# ==========================================
def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def inspect_model_freeze(model):
    """檢查模型哪些層被凍結，哪些層是可訓練的"""
    print_separator("Model Parameter Status")
    trainable_params = 0
    frozen_params = 0
    
    print(f"{'Module Name':<40} | {'Status':<10}")
    print("-" * 55)
    
    # 檢查主要組件
    components = [
        ("image_encoder (ViT)", model.image_encoder),
        ("prompt_encoder", model.prompt_encoder),
        ("mask_decoder", model.mask_decoder),
        ("fusion_module", model.fusion_module),
        ("gate_module", model.gate_module),
        ("text_encoder (CLIP)", model.text_encoder),
        ("pe_layer (Positional)", model.pe_layer)
    ]

    for name, submodule in components:
        # 對於 Tensor (如 pe_layer)
        if isinstance(submodule, torch.nn.Parameter):
            requires_grad = submodule.requires_grad
        # 對於 Module
        else:
            # 抽樣檢查第一個參數
            try:
                requires_grad = next(submodule.parameters()).requires_grad
            except StopIteration:
                requires_grad = False # 無參數層

        status = "🟢 Train" if requires_grad else "❄️ Freeze"
        print(f"{name:<40} | {status}")

    # 統計總參數量
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            frozen_params += p.numel()
            
    print("-" * 55)
    print(f"Total Trainable Params: {trainable_params / 1e6:.2f} M")
    print(f"Total Frozen Params:    {frozen_params / 1e6:.2f} M")

# ==========================================
# 3. 主測試邏輯
# ==========================================
def run_sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Running on device: {device}")
    
    # --- Step 1: 建立模型 ---
    print_separator("Building Model")
    try:
        model = build_weather_sam_vit_h(checkpoint=None)
        model.to(device)
        print("✅ Model built successfully.")
    except Exception as e:
        print(f"❌ Model build failed: {e}")
        return

    # --- Step 2: 初始化訓練器 ---
    print_separator("Initializing Trainer")
    
    mock_ds = MockWeatherDataset(length=128, image_size=1024)
    mock_loader = DataLoader(mock_ds, batch_size=4, collate_fn=collate_fn)
    
    try:
        trainer = WeatherSAMTrainer(
            model=model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device=device,
            lr=1e-4
        )
        print("✅ Trainer initialized.")
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        return

    inspect_model_freeze(model)

    # --- Step 3 & 4: Forward + Loss + Backward (全部在 Autocast 保護下) ---
    print_separator("Testing Pipeline (Forward -> Loss -> Backward)")
    
    batch = next(iter(mock_loader))
    
    # 準備輸入
    batched_input = []
    batch_size = len(batch['text_prompts'])
    for i in range(batch_size):
        batched_input.append({
            'image': batch['image'][i].to(device),
            'reference_mask': batch['reference_mask'][i].to(device),
            'text_prompts': batch['text_prompts'][i],
            'original_size': batch['original_size'][i]
        })
    gt_masks = batch['gt_mask'].to(device)

    # 監控記憶體
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"🧠 GPU Memory (Pre-Forward): {mem_before:.2f} MB")

    try:
        # ★★★ 關鍵修改：將 Loss 計算也包進 autocast ★★★
        with torch.amp.autocast('cuda' if device=='cuda' else 'cpu'):
            print("▶️  Executing Forward Pass...")
            outputs = model(batched_input, multimask_output=True)
            print("✅ Forward pass successful.")

            print("▶️  Calculating Loss...")
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
            
            # 針對 Batch 中第一張圖計算
            i = 0
            low_res = outputs[i]['low_res_logits'] 
            
            # 使用官方的 postprocess 放大
            full_res_logits = model.postprocess_masks(
                low_res,
                input_size=(1024, 1024),
                original_size=(1024, 1024)
            )
            
            # 取最佳 mask index
            best_idx = torch.argmax(outputs[i]['iou_predictions'], dim=1)
            selected_logits = full_res_logits[torch.arange(19), best_idx, :, :].unsqueeze(0)
            
            # 通過 Fusion Head
            fused_logits = model.semantic_fusion_head(selected_logits)
            
            # 確保 gt 的類別是 long
            gt_mask_var = gt_masks[i].unsqueeze(0).long()
            
            loss_val = loss_fn(fused_logits, gt_mask_var)
            
            print(f"📉 Calculated Loss: {loss_val.item():.4f}")

        # Backward Pass (雖然 backward 通常可以放在外面，但為了保險起見，scale 必須接續上面的 loss)
        print("▶️  Executing Backward Pass...")
        trainer.optimizer.zero_grad()
        trainer.scaler.scale(loss_val).backward()
        print("✅ Backward pass successful (Gradients computed).")
        
        # 檢查梯度
        print("🔍 Checking Gradients on Semantic Fusion Head:")
        has_grad = False
        for name, param in model.semantic_fusion_head.named_parameters():
            if param.grad is not None:
                print(f"   Layer: {name} | Grad Mean: {param.grad.mean().item():.2e} | ✅ Has Grad")
                has_grad = True
                break
        
        if not has_grad:
            print("⚠️ Warning: No gradients found in Semantic Fusion Head!")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if device == "cuda":
        mem_after = torch.cuda.max_memory_allocated() / 1024**2
        print(f"🧠 Peak GPU Memory: {mem_after:.2f} MB")

    print_separator("Sanity Check Complete")
    print("🎉 System looks healthy! You are ready for real training.")

if __name__ == "__main__":
    run_sanity_check()