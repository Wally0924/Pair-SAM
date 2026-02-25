# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        MaskDecoder 用於根據融合後的特徵圖 (Fused Features) 和提示 (Prompts) 預測遮罩。

        Args:
          transformer_dim (int): Transformer 的通道維度 (通常為 256)。
          transformer (nn.Module): 用於預測遮罩的 TwoWayTransformer。
          num_multimask_outputs (int): 預測的多義性遮罩數量 (通常為 3)。
          activation (nn.Module): 上採樣層使用的激活函數。
          iou_head_depth (int): IoU 預測頭的 MLP 深度。
          iou_head_hidden_dim (int): IoU 預測頭的隱藏層維度。
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # IoU Token: 用於預測遮罩品質
        self.iou_token = nn.Embedding(1, transformer_dim)
        # Mask Tokens: 用於生成遮罩的 Embeddings (包含 1 個最優遮罩 + 3 個多義性遮罩)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 輸出上採樣層: 將 Transformer 的輸出 (64x64) 上採樣回原圖比例 (通常放大 4 倍至 256x256)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        # Hyper-networks: 每個 Mask Token 對應一個 MLP，用於生成動態卷積權重
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # IoU Prediction Head: 預測每個遮罩的 IoU 分數
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          image_embeddings (torch.Tensor): 來自 Fusion Module 的特徵圖 F_fuse。
                                           Shape: (B, 256, 64, 64)
          image_pe (torch.Tensor): 位置編碼 (Positional Encoding)。
                                   Shape: (B, 256, 64, 64) 或 (1, 256, 64, 64)
          sparse_prompt_embeddings (torch.Tensor): 來自 Prompt Encoder 的文字提示特徵。
                                                   Shape: (B, N_prompts, 256)
          dense_prompt_embeddings (torch.Tensor): 來自 Prompt Encoder 的遮罩提示特徵。
                                                  Shape: (B, 256, 64, 64)
          multimask_output (bool): 是否輸出多個層次的遮罩。

        Returns:
          masks (torch.Tensor): 預測的遮罩 (B, C, H, W)
          iou_pred (torch.Tensor): 預測的 IoU 分數 (B, C)
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 根據是否需要多遮罩輸出來切片選擇
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """核心預測邏輯 [cite: 100, 121, 294]"""
        
        # 1. 準備 Output Tokens (IoU Token + Mask Tokens)
        # Shape: (1, 1+num_mask_tokens, 256) -> 擴展至 Batch Size -> (B, N_tokens, 256)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        
        # 2. 拼接 Prompt Tokens
        # 將 Output Tokens 與文字提示 (Sparse Prompts) 串接
        # Shape: (B, N_tokens + N_prompts, 256)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 3. 處理影像特徵 (Fused Features) - 關鍵修改部分
        # 檢查 Batch Size 是否匹配。
        # 如果 image_embeddings 的 batch size 與 tokens (prompts) 的 batch size 不同 (通常是 1 vs B)，
        # 代表是「單圖多Prompt」模式，需要複製影像特徵。
        # 如果相同 (B vs B)，代表是「Batch Training」模式，直接使用，避免錯誤複製。
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        # 加上 Dense Mask Prompts (通常來自 WeatherPromptEncoder 的 mask embeddings)
        src = src + dense_prompt_embeddings
        
        # 4. 處理位置編碼 (Positional Encoding)
        # 同樣邏輯：如果 PE 只有一份 (1, ...)，則擴展；如果是 Batch (B, ...)，則直接用。
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe

        b, c, h, w = src.shape

        # 5. 執行 Two-Way Transformer
        # hs: 經過 Transformer 的 Tokens (IoU + Masks)
        # src: 經過 Transformer 的影像特徵 (已融合 Prompt 資訊)
        hs, src = self.transformer(src, pos_src, tokens)
        
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 6. 上採樣與遮罩生成
        # 將 Transformer 輸出的特徵圖轉置回 (B, C, H, W) 並上採樣
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            # 使用 Mask Tokens 生成動態 MLP 的權重
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        
        b, c, h, w = upscaled_embedding.shape
        # 矩陣乘法生成最終遮罩 (B, Num_Masks, H, W)
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 7. 預測 IoU 品質
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x