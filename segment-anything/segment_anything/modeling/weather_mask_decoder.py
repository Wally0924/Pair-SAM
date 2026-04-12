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
        num_classes: int = 19,
    ) -> None:
        """
        MaskDecoder 用於根據融合後的特徵圖 (Fused Features) 和提示 (Prompts) 預測遮罩。

        Args:
          transformer_dim (int): Transformer 的通道維度 (通常為 256)。
          transformer (nn.Module): 用於預測遮罩的 TwoWayTransformer。
          num_multimask_outputs (int): 預測的多義性遮罩數量 (通常為 3，保留供 SAM 原始 forward 使用)。
          activation (nn.Module): 上採樣層使用的激活函數。
          iou_head_depth (int): IoU 預測頭的 MLP 深度。
          iou_head_hidden_dim (int): IoU 預測頭的隱藏層維度。
          num_classes (int): 語意分割類別數 (Mask2Former 模式使用)。
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_classes = num_classes

        # ── 原始 SAM tokens（保留，供向後兼容） ──
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 輸出上採樣層: 將 Transformer 的輸出 (64x64) 上採樣回原圖比例 (256x256)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 原始 Hyper-networks（保留，供向後兼容）
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )

        # 原始 IoU Prediction Head（保留，供向後兼容）
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
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

        # 每個類別的專屬 IoU query token
        self.class_iou_tokens = nn.Embedding(num_classes, transformer_dim)
        nn.init.normal_(self.class_iou_tokens.weight, std=0.02)

        # 每個類別的專屬 hypernetwork MLP（對應 Mask2Former 的 per-query mask head）
        self.class_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(num_classes)]
        )

        # 每個類別的專屬 IoU 預測頭（輸出 1 個分數，不再是 3 個 candidates）
        self.class_iou_head = MLP(transformer_dim, iou_head_hidden_dim, 1, iou_head_depth)

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

        # 1. 各類別專屬 mask query tokens（class_iou_tokens 已移除）
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