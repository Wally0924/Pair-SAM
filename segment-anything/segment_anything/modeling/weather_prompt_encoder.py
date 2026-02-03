# import torch
# import torch.nn as nn
# from typing import Tuple, Optional, Type
# from .common import LayerNorm2d
# from .location_encoder import LocationEncoder

# class WeatherPromptEncoder(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         image_embedding_size: Tuple[int, int],
#         input_image_size: Tuple[int, int],
#         mask_in_chans: int,
#         activation: Type[nn.Module] = nn.GELU,
#     ) -> None:
#         """
#         混合式 Prompt Encoder：
#         1. 接收 Text Embeddings 作為 Sparse Prompt
#         2. 接收 Location Embeddings 作為 Sparse Prompt
#         2. 接收 Mask Inputs 作為 Dense Prompt (需經過 Downscaling)
        
#         Args:
#             embed_dim (int): 特徵維度 (SAM預設 256)。
#             image_embedding_size (tuple): Image Encoder 輸出的特徵圖尺寸 (64, 64)。
#             input_image_size (tuple): 原始輸入影像尺寸 (1024, 1024)。
#             mask_in_chans (int): Mask Downscaling 的中間層通道數 (預設 16)。
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.input_image_size = input_image_size
#         self.image_embedding_size = image_embedding_size

#         # ==============================================================================
#         # Part 1: No Mask Embedding (當沒有 Mask 輸入時的預設值)
#         # ==============================================================================
#         self.no_mask_embed = nn.Embedding(1, embed_dim)

#         # ==============================================================================
#         # Part 2: Mask Downscaling (把遮罩提示加回來)
#         # 這是從原本 SAM 的 prompt_encoder.py 複製過來的邏輯 。
#         # 它負責將輸入的低解析度 Mask (通常是 256x256 或 原圖大小) 
#         # 下採樣 4 倍並編碼為 embed_dim (256)。
#         # ==============================================================================
#         self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        
#         self.mask_downscaling = nn.Sequential(
#             nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
#             LayerNorm2d(mask_in_chans // 4),
#             activation(),
#             nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
#             LayerNorm2d(mask_in_chans),
#             activation(),
#             nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
#         )

#     def _get_device(self) -> torch.device:
#         return self.no_mask_embed.weight.device

#     def forward(
#         self, 
#         text_embeddings: torch.Tensor, 
#         mask_inputs: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             text_embeddings (torch.Tensor): 來自 Text Encoder 的特徵 (Sparse)。
#                                             Shape: (B, Num_Prompts, embed_dim)
#             mask_inputs (torch.Tensor, optional): 遮罩提示 (Dense)。
#                                                   Shape: (B, 1, 4*H_feat, 4*W_feat) 
#                                                   SAM 預設輸入通常是 256x256 (即 4x64)。

#         Returns:
#             sparse_embeddings (torch.Tensor): (B, K, 256)
#             dense_embeddings (torch.Tensor):  (B, 256, 64, 64)
#         """
#         bs = text_embeddings.shape[0]

#         # 1. Sparse Embeddings (直接使用文字特徵)
#         sparse_embeddings = text_embeddings

#         # 2. Dense Embeddings (處理遮罩)
#         if mask_inputs is not None:
#             # 如果有輸入 Mask，通過 CNN 進行編碼 [cite: 198]
#             dense_embeddings = self.mask_downscaling(mask_inputs)
#         else:
#             # 如果沒有輸入 Mask，使用 no_mask_embed 並廣播到全圖 [cite: 178]
#             dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
#                 bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
#             )

#         return sparse_embeddings, dense_embeddings


import torch
import torch.nn as nn
from typing import Tuple, Optional, Type
from .location_encoder import LocationEncoder 

class WeatherPromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int, # 為了保持參數接口兼容保留，但內部不再使用
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        修改版 WeatherPromptEncoder：
        1. [啟用] 接收 Text Embeddings (Sparse Prompt)
        2. [啟用] 接收 Location Embeddings (Sparse Prompt, 基於 GeoCLIP)
        3. [停用] Mask Inputs (Dense Prompt) 
           -> 因為 Reference Mask 已經在 Fusion Module 中與影像特徵融合，
              這裡不再重複輸入，避免強硬的空間引導與 Fusion 效果衝突。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # ==============================================================================
        # Part 1: No Mask Embedding (Dense Prompt 的固定輸出)
        # ==============================================================================
        # 當不提供具體的 Mask Prompt 時，SAM 使用一個可學習的 Embedding 廣播到全圖
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def _get_device(self) -> torch.device:
        return self.no_mask_embed.weight.device

    def forward(
        self, 
        text_embeddings: torch.Tensor, 
        location_embeddings: Optional[torch.Tensor] = None,
        mask_inputs: Optional[torch.Tensor] = None, # 雖然保留此參數以相容介面，但內部會忽略
        location_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embeddings: (B, K, embed_dim) 來自 TextEncoder
            location_embeddings: (B, K, embed_dim) 來自 LocationEncoder (已擴展)
            mask_inputs: (Ignored)

        Returns:
            sparse_embeddings (torch.Tensor): 結合了 Text 與 Location 的 Prompts
                                              Shape: (B, K + 1, 256) (若有 Location)
            dense_embeddings (torch.Tensor):  固定為 no_mask_embed
                                              Shape: (B, 256, 64, 64)
        """
        bs = text_embeddings.shape[0]

        # 1. 處理 Sparse Embeddings (Text + Location)
        sparse_embeddings = text_embeddings # (B, K, 256)

        if location_embeddings is not None:
            # 確保維度匹配，直接串接
            # text: (K, 1, 256), location: (K, 1, 256) -> cat -> (K, 2, 256)
            # 注意：這裡假設傳入前已經做過維度處理 (unsqueeze & repeat)
            sparse_embeddings = torch.cat([sparse_embeddings, location_embeddings], dim=1)

        # 2. 處理 Dense Embeddings (固定策略)
        # 不論是否有傳入 mask_inputs，我們都忽略它，直接使用 no_mask_embed。
        # 這等於告訴 Mask Decoder：「沒有額外的形狀提示，請依賴輸入的 Image Embeddings (Fusion後的結果)」。
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )

        return sparse_embeddings, dense_embeddings