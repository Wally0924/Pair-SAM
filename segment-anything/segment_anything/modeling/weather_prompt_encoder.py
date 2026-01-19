import torch
import torch.nn as nn
from typing import Tuple, Optional, Type
from .common import LayerNorm2d

class WeatherPromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        混合式 Prompt Encoder：
        1. 接收 Text Embeddings 作為 Sparse Prompt
        2. 接收 Mask Inputs 作為 Dense Prompt (需經過 Downscaling)
        
        Args:
            embed_dim (int): 特徵維度 (SAM預設 256)。
            image_embedding_size (tuple): Image Encoder 輸出的特徵圖尺寸 (64, 64)。
            input_image_size (tuple): 原始輸入影像尺寸 (1024, 1024)。
            mask_in_chans (int): Mask Downscaling 的中間層通道數 (預設 16)。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # ==============================================================================
        # Part 1: No Mask Embedding (當沒有 Mask 輸入時的預設值)
        # ==============================================================================
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # ==============================================================================
        # Part 2: Mask Downscaling (把遮罩提示加回來)
        # 這是從原本 SAM 的 prompt_encoder.py 複製過來的邏輯 。
        # 它負責將輸入的低解析度 Mask (通常是 256x256 或 原圖大小) 
        # 下採樣 4 倍並編碼為 embed_dim (256)。
        # ==============================================================================
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def _get_device(self) -> torch.device:
        return self.no_mask_embed.weight.device

    def forward(
        self, 
        text_embeddings: torch.Tensor, 
        mask_inputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embeddings (torch.Tensor): 來自 Text Encoder 的特徵 (Sparse)。
                                            Shape: (B, Num_Prompts, embed_dim)
            mask_inputs (torch.Tensor, optional): 遮罩提示 (Dense)。
                                                  Shape: (B, 1, 4*H_feat, 4*W_feat) 
                                                  SAM 預設輸入通常是 256x256 (即 4x64)。

        Returns:
            sparse_embeddings (torch.Tensor): (B, K, 256)
            dense_embeddings (torch.Tensor):  (B, 256, 64, 64)
        """
        bs = text_embeddings.shape[0]

        # 1. Sparse Embeddings (直接使用文字特徵)
        sparse_embeddings = text_embeddings

        # 2. Dense Embeddings (處理遮罩)
        if mask_inputs is not None:
            # 如果有輸入 Mask，通過 CNN 進行編碼 [cite: 198]
            dense_embeddings = self.mask_downscaling(mask_inputs)
        else:
            # 如果沒有輸入 Mask，使用 no_mask_embed 並廣播到全圖 [cite: 178]
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings