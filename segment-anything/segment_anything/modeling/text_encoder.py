import torch
import torch.nn as nn
import clip

class TextEncoder(nn.Module):
    def __init__(
        self, 
        model_name: str = "ViT-B/32", 
        output_dim: int = 256,
        freeze: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Module 3: Text Encoder (OpenAI CLIP version)
        
        Args:
            model_name: CLIP 模型版本 (e.g., "ViT-B/32", "ViT-L/14")
            output_dim: SAM 的特徵維度 (通常是 256)
            freeze: 是否凍結 CLIP 參數 (論文 P.19 建議凍結 )
        """
        super().__init__()
        self.device = device
        
        # 1. 載入 OpenAI CLIP 模型
        # jit=False 讓模型保持 PyTorch 格式，方便後續微調或儲存
        self.clip_model, _ = clip.load(model_name, device=device, jit=False)
        
        # 取得 CLIP 的輸出維度 (ViT-B/32 為 512, ViT-L/14 為 768)
        # text_projection 是 CLIP 內部的投影層，我們取它的輸出維度
        clip_embed_dim = self.clip_model.text_projection.shape[1]
        
        # 2. 投影層: 將 CLIP 特徵 (512) 轉為 SAM 特徵 (256)
        self.projection = nn.Linear(clip_embed_dim, output_dim)
        
        # 3. 凍結權重
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            # 注意: projection 層是我們新加的，預設不凍結，讓它學習如何對齊
            
    def forward(self, text_list: list[str]) -> torch.Tensor:
        """
        Args:
            text_list: 文字提示列表，例如 ["road", "car"]
        Returns:
            text_embeddings: Shape (B, 1, 256)
        """
        # 1. Tokenization (將文字轉為 ID)
        # clip.tokenize 會自動處理 padding 和 truncation
        text_tokens = clip.tokenize(text_list).to(next(self.clip_model.parameters()).device)
        
        # 2. CLIP Encoding
        # encode_text 會輸出經過正規化的特徵
        with torch.no_grad(): # 凍結 CLIP 梯度
            text_features = self.clip_model.encode_text(text_tokens) # (B, 512)
            
        # 3. 正規化 (再次確保數值穩定)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 4. 投影到 SAM 維度 (512 -> 256)
        # 轉為 float (CLIP 有時預設是用 float16)
        embeddings = self.projection(text_features.float()) # (B, 256)
        
        # 5. 增加維度以符合 Prompt 格式 (B, 1, 256)
        return embeddings.unsqueeze(1)