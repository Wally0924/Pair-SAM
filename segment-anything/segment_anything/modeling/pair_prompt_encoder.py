# pair_prompt_encoder.py

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PairPromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
    ) -> None:
        """
        修改版 PairPromptEncoder (Sparse Prompt Strategy):
        1. 接收 Text Embeddings
        2. 接收 Location Embeddings
        將兩者串接作為 Transformer 的 Tokens；dense prompt 固定為 no_mask_embed。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # No Mask Embedding (Dense Prompt 的預設值)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def _get_device(self) -> torch.device:
        return self.no_mask_embed.weight.device

    def forward(
        self,
        text_embeddings: torch.Tensor,
        location_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embeddings: (B, K, 256) - 文字 Token
            location_embeddings: (B, K, 256) - 地理位置 Token (已在 SAM 中 repeat)
            
        Returns:
            sparse_embeddings: (B, K + 1, 256) - 串接後的 Token 序列
            dense_embeddings: (B, 256, 64, 64) - 固定為 no_mask_embed
        """
        bs = text_embeddings.shape[0]

        # 1. 處理 Sparse Embeddings
        sparse_embeddings = text_embeddings # (B, K, 256)

        if location_embeddings is not None:
            # 串接 Text 和 Location
            # text: (K, 1, 256)
            # location: (K, 1, 256)
            # Result: (K, 2, 256) -> 第一個 Token 是 Text，第二個是 Location
            sparse_embeddings = torch.cat([sparse_embeddings, location_embeddings], dim=1)

        # 2. 處理 Dense Embeddings (固定策略)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )

        return sparse_embeddings, dense_embeddings