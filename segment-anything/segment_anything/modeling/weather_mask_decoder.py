# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Type

from .common import LayerNorm2d

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        num_classes: int = 19,
    ) -> None:
        """
        MaskDecoder — Mask2Former-style 語意分割解碼器。

        Args:
          transformer_dim (int): Transformer 的通道維度 (通常為 256)。
          transformer (nn.Module): 用於預測遮罩的 TwoWayTransformer。
          activation (nn.Module): 上採樣層使用的激活函數。
          num_classes (int): 語意分割類別數。
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes

        # 輸出上採樣層: 將 Transformer 的輸出 (64x64) 上採樣回原圖比例 (256x256)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # ══════════════════════════════════════════════════════
        # [Mask2Former-style] 19 個語意類別專屬模組
        # 學術依據：Mask2Former (Cheng et al., CVPR 2022)
        #   — 每個類別有獨立的 learnable query token
        #   — 所有 K 個 class query 在同一個 transformer sequence 中互相 attend
        #   — 每個類別有專屬的 hypernetwork MLP 生成動態卷積權重
        # 相較於共享 4 個 mask_tokens 的原始 SAM 設計：
        #   — 各類別不再互相覆蓋初始化偏差
        #   — cross-class self-attention 使類別間具備互斥感知能力
        # ══════════════════════════════════════════════════════

        # 每個類別的專屬 mask query token（對應 Mask2Former 的 object queries）
        self.class_mask_tokens = nn.Embedding(num_classes, transformer_dim)
        nn.init.normal_(self.class_mask_tokens.weight, std=0.02)

        # 每個類別的專屬 hypernetwork MLP（對應 Mask2Former 的 per-query mask head）
        self.class_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(num_classes)]
        )

        # [ablation] 'unified' = 所有 class query 同序列（跨類別 self-attention）；
        # 'per_class' = 每類別獨立 forward（移除跨類別交互）。預設 unified = FULL 行為。
        self.decoder_mode = 'unified'

    def forward_semantic(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        class_ids: List[int],
    ) -> torch.Tensor:
        """
        [Mask2Former-style] 語意分割專用 forward。

        所有 K 個 active class query 在同一個 transformer sequence 中一起處理，
        透過 self-attention 實現跨類別互斥感知。每個類別輸出 1 個 mask（不再有 3 個 candidates）。

        Args:
          image_embeddings: (1, 256, 64, 64)
          image_pe:          (1, 256, 64, 64)
          sparse_prompt_embeddings: (K, N_tok, 256)  K 個類別、每類別 N_tok 個 prompt token
          dense_prompt_embeddings:  (1, 256, 64, 64)
          class_ids: List[int], len=K — 本張圖 active 的類別 ID

        Returns:
          masks: (1, K, 256, 256) — 每個類別 1 張 logit map
        """
        if self.decoder_mode == 'per_class':
            return self.predict_masks_per_class(
                image_embeddings, image_pe,
                sparse_prompt_embeddings, dense_prompt_embeddings,
                class_ids,
            )
        return self.predict_masks_semantic(
            image_embeddings, image_pe,
            sparse_prompt_embeddings, dense_prompt_embeddings,
            class_ids,
        )

    def predict_masks_semantic(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        class_ids: List[int],
    ) -> torch.Tensor:
        """
        核心語意解碼邏輯（Mask2Former-style，單次 transformer forward）。

        移除 IoU prediction：新架構每類別只有 1 個 mask，無候選選擇需求，
        IoU head 無任何下游用途，因此從 sequence 和 loss 中一併移除。
        """
        K = len(class_ids)
        dev = image_embeddings.device
        class_ids_t = torch.tensor(class_ids, dtype=torch.long, device=dev)

        # 1. 各類別專屬 mask query tokens
        class_q = self.class_mask_tokens(class_ids_t)  # (K, 256)

        # 2. 統一 token 序列：[K 個 class mask query, K*N_tok 個 prompt token]
        # 所有類別 query 在同一 sequence，TwoWayTransformer self-attention 使它們互相感知
        output_tokens = class_q.unsqueeze(0)  # (1, K, 256)

        N_tok = sparse_prompt_embeddings.shape[1]
        prompt_tokens = sparse_prompt_embeddings.reshape(1, K * N_tok, self.transformer_dim)

        tokens = torch.cat([output_tokens, prompt_tokens], dim=1)  # (1, K + K*N_tok, 256)

        # 3. 影像特徵 + dense prompt
        # dense_prompt_embeddings 可能是 (K, 256, H, W)（來自 PromptEncoder batch=K），
        # 取第一個（no_mask_embed 對所有 K 類別完全相同），確保 src batch=1
        dp = dense_prompt_embeddings[:1]  # (1, 256, H, W)
        src = image_embeddings + dp       # (1, 256, 64, 64)
        pos_src = image_pe
        b, c, h, w = src.shape

        # 4. 單次 transformer forward
        hs, src = self.transformer(src, pos_src, tokens)  # hs: (1, K + K*N_tok, 256)

        # 5. 提取各類別 mask token 輸出
        mask_token_out = hs[:, :K, :]  # (1, K, 256)

        # 6. 上採樣共享影像特徵
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled = self.output_upscaling(src)  # (1, 32, 256, 256)
        bu, cu, hu, wu = upscaled.shape

        # 7. 各類別專屬 hypernetwork 生成 mask
        masks = []
        for i, cls_id in enumerate(class_ids):
            w_hyper = self.class_hypernetworks_mlps[cls_id](mask_token_out[:, i, :])  # (1, 32)
            mask_i  = (w_hyper @ upscaled.view(bu, cu, hu * wu)).view(1, 1, hu, wu)
            masks.append(mask_i)

        return torch.cat(masks, dim=1)  # (1, K, 256, 256)

    def predict_masks_per_class(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        class_ids: List[int],
    ) -> torch.Tensor:
        """[ablation] 逐類別獨立解碼：每個 class 以 K=1 單獨跑一次 transformer，
        移除跨類別 self-attention。複用 predict_masks_semantic，參數量與統一查詢版相同。
        """
        masks = []
        for i, cls_id in enumerate(class_ids):
            mask_i = self.predict_masks_semantic(
                image_embeddings, image_pe,
                sparse_prompt_embeddings[i:i + 1],   # (1, N_tok, 256)
                dense_prompt_embeddings,
                [cls_id],
            )  # (1, 1, 256, 256)
            masks.append(mask_i)
        return torch.cat(masks, dim=1)  # (1, K, 256, 256)


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