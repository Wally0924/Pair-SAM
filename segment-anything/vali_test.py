import torch
import torch.nn.functional as F
import gc
from segment_anything.build_weather_sam import build_weather_sam_vit_b
# 若要測試 ViT-H，請解開下面這行
from segment_anything.build_weather_sam import build_weather_sam_vit_h 
from utils.new_loss import SAMLoss

def verify_pipeline():
    # 0. 清理 GPU 記憶體 (防止之前的殘留)
    torch.cuda.empty_cache()
    gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    # 1. 建立模型 (建議先用 ViT-B 跑通流程)
    print("🏗️ Building Model (ViT-H)...")
    try:
        model = build_weather_sam_vit_h(checkpoint=None) 
        model.to(device)
    except Exception as e:
        print(f"❌ Model Build Failed: {e}")
        return

    # ★★★ 關鍵修改：手動凍結骨幹網路 ★★★
    # 這步能節省約 70%-80% 的顯存，因為不需要儲存巨大的反向傳播圖
    print("❄️ Freezing Image & Text Encoders...")
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        
    # 確保其他模組是可訓練的 (Mask Decoder, Fusion, etc.)
    # 這樣我們才能測試 backward() 是否正常運作
    model.train() 

    # 2. 模擬輸入資料 (改為 Batch Size = 1 以求穩)
    print("🎲 Generating Dummy Data (Batch Size = 1)...")
    B = 2 
    C, H, W = 3, 1024, 1024
    
    dummy_images = torch.randn(B, C, H, W).to(device)
    dummy_ref_masks = torch.randn(B, C, H, W).to(device) # RGB Mask
    dummy_gt_masks = torch.randint(0, 19, (B, H, W)).long().to(device)
    
    # 模擬 Prompts
    dummy_prompts = [["car", "road"],["tree", "building"]] 
    dummy_original_sizes = [(1080, 1920), (1080, 1920)]

    batched_input = []
    for i in range(B):
        batched_input.append({
            'image': dummy_images[i],
            'reference_mask': dummy_ref_masks[i],
            'text_prompts': dummy_prompts[i],
            'original_size': dummy_original_sizes[i]
        })

    # 3. 初始化 Loss
    criterion = SAMLoss(focal_weight=20.0, dice_weight=1.0, iou_weight=1.0)
    
    # 只優化可訓練的參數
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"   Trainable Parameters: {len(trainable_params)} tensors")
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # 4. 前向傳播測試
    print("🚀 Running Forward Pass...")
    try:
        # 使用混合精度 (AMP) 進一步省記憶體
        with torch.amp.autocast('cuda'):
            # 這裡我們用 torch.no_grad() 包住 Image Encoder 的 forward 嗎？
            # 不需要，因為上面已經設了 requires_grad=False，PyTorch 知道不需要建圖
            
            outputs = model(batched_input, multimask_output=True)
            
            print(f"   Output Count: {len(outputs)}")
            print(f"   Mask Shape: {outputs[0]['masks'].shape}")

            # 5. Loss 計算
            print("📉 Calculating Loss...")
            total_loss = 0
            for i in range(B):
                full_res_logits = F.interpolate(
                    outputs[i]['low_res_logits'],
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False
                )
                
                loss, loss_dict = criterion(
                    pred_masks=full_res_logits,
                    gt_mask=dummy_gt_masks[i],
                    iou_predictions=outputs[i]['iou_predictions'],
                    text_prompts=batched_input[i]['text_prompts']
                )
                total_loss += loss
                print(f"   Loss Value: {loss.item():.4f}")

        # 6. 反向傳播測試
        print("🔙 Running Backward Pass...")
        optimizer.zero_grad()
        
        # 使用 Scaler 模擬真實訓練情境
        scaler = torch.amp.GradScaler('cuda')
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ Verify Pipeline Passed Successfully!")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    except RuntimeError as e:
        print(f"❌ Runtime Error: {e}")
        if "out of memory" in str(e):
            print("💡 Critical: Still OOM. Try setting 'input_image_size' smaller in build_weather_sam.py temporarily.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ General Error: {e}")

if __name__ == "__main__":
    verify_pipeline()