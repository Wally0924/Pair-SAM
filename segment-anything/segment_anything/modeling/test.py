import torch
import numpy as np

# 匯入我們定義的模組
from text_encoder import TextEncoder
from weather_prompt_encoder import WeatherPromptEncoder

def test_text_and_prompt_integration():
    print("=== 開始測試 TextEncoder 與 WeatherPromptEncoder 整合 ===")
    
    # 1. 設定裝置 (檢查是否有 GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")

    # ==========================================
    # 步驟 1: 初始化模型
    # ==========================================
    print("\n[Step 1] 初始化模型...")
    
    # A. 初始化 Text Encoder (使用 CLIP)
    # output_dim=256 是為了對齊 SAM 的特徵維度
    text_encoder = TextEncoder(model_name="ViT-B/32", output_dim=256, device=device)
    
    # B. 初始化 Weather Prompt Encoder
    # 這些參數是對應 SAM ViT-H 的標準設定
    prompt_encoder = WeatherPromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    ).to(device)

    # ==========================================
    # 步驟 2: 準備模擬資料
    # ==========================================
    print("\n[Step 2] 準備輸入資料...")
    
    # 假設 Batch Size = 2
    batch_size = 2
    
    # A. 文字提示 (List of strings)
    text_prompts = ["road surface", "traffic signs"]
    print(f"輸入文字: {text_prompts}")

    # B. 幾何提示 (模擬點擊提示)
    # 格式: (座標, 標籤)
    # 座標 shape: (B, N, 2) -> 這裡是 (2, 1, 2) 每個 batch 1 個點
    point_coords = torch.tensor([[[500, 500]], [[200, 300]]], dtype=torch.float, device=device)
    # 標籤 shape: (B, N) -> 1 代表前景點
    point_labels = torch.tensor([[1], [1]], dtype=torch.int, device=device)
    
    points = (point_coords, point_labels)

    # ==========================================
    # 步驟 3: 執行 Text Encoder
    # ==========================================
    print("\n[Step 3] 執行 Text Encoder...")
    
    # 輸入: ["road surface", "traffic signs"]
    # 輸出: Tensor (B, 1, 256)
    text_embeddings = text_encoder(text_prompts)
    
    print(f"Text Embeddings Shape: {text_embeddings.shape}")
    # 預期: torch.Size([2, 1, 256])
    
    # ==========================================
    # 步驟 4: 執行 Prompt Encoder (整合)
    # ==========================================
    print("\n[Step 4] 執行 Prompt Encoder...")
    
    # 我們同時傳入 "幾何點 (points)" 和 "文字特徵 (text_embeddings)"
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=points,
        boxes=None,
        masks=None,
        text_embeddings=text_embeddings # 這是我們新增的參數
    )

    # ==========================================
    # 步驟 5: 檢查輸出結果
    # ==========================================
    print("\n[Step 5] 驗證輸出維度...")
    
    # 1. 檢查 Sparse Embeddings (稀疏提示)
    # 預期維度: (B, N_tokens, 256)
    # N_tokens = (幾何點數量) + (文字數量)
    # 在這個例子: 1個點 + 1個文字 = 2個 tokens
    expected_tokens = point_coords.shape[1] + 1 
    
    print(f"Sparse Embeddings Shape: {sparse_embeddings.shape}")
    
    if sparse_embeddings.shape == (batch_size, expected_tokens, 256):
        print(f"✅ Sparse Embeddings 通過測試! (Token數: {sparse_embeddings.shape[1]})")
        print(f"   - 包含: {point_coords.shape[1]} 個點 + 1 個文字特徵")
    else:
        print(f"❌ Sparse Embeddings 維度錯誤，預期 (2, {expected_tokens}, 256)")

    # 2. 檢查 Dense Embeddings (稠密提示 - 用於 Mask)
    # 如果沒傳入 mask，應該是 "no_mask_embed" 的廣播
    print(f"Dense Embeddings Shape: {dense_embeddings.shape}")
    if dense_embeddings.shape == (batch_size, 256, 64, 64):
        print("✅ Dense Embeddings 通過測試!")
    else:
        print("❌ Dense Embeddings 維度錯誤")

    print("\n=== 整合測試完成 ===")

if __name__ == "__main__":
    test_text_and_prompt_integration()