import torch
import torch.nn as nn
from typing import Optional, Tuple

from .prompt_encoder import PromptEncoder

class WeatherPromptEncoder(PromptEncoder):
    def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans):
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text_embeddings: Optional[torch.Tensor] = None, # 新增這個參數
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        擴充版 Forward，支援 Text Embeddings 輸入。
        """
        
        # 1. 先呼叫原始的 forward 處理點、框、遮罩
        # sparse_embeddings: (B, N_original, 256)
        # dense_embeddings: (B, 256, 64, 64)
        sparse_embeddings, dense_embeddings = super().forward(points, boxes, masks)
        
        # 2. 處理 Text Embeddings
        if text_embeddings is not None:
            # text_embeddings shape 預期為 (B, K, 256)
            # 其中 K 是文字提示的數量 (通常是 1 或多個)
            
            # 檢查 Batch Size 是否一致 (Optional debug)
            if text_embeddings.shape[0] != sparse_embeddings.shape[0]:
                # 如果原本沒有任何 point/box prompt，sparse_embeddings batch 可能會是預設值
                # 這裡可能需要簡單的廣播或檢查機制，視你的 dataloader 而定
                pass

            # 3. 將文字特徵串接到 Sparse Embeddings (作為額外的 Prompt Token)
            # 結果 shape: (B, N_original + K, 256)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        return sparse_embeddings, dense_embeddings