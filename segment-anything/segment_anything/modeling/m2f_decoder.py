# ============================================================================
# Vendored from: facebookresearch/Mask2Former
#   Files:
#     mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
#       (SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP,
#        MultiScaleMaskedTransformerDecoder)
#     mask2former/modeling/transformer_decoder/position_encoding.py
#       (PositionEmbeddingSine — 原始出處為 facebookresearch/detr, Apache-2.0)
#   Commit: 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a
#   License: MIT (Copyright (c) Facebook, Inc. and its affiliates.)
#   Paper: Cheng et al., "Masked-attention Mask Transformer for Universal
#          Image Segmentation" (Mask2Former), CVPR 2022. arXiv:2112.01527
#
# [WeatherSAM adaptations]（完整清單；其餘逐行同上游）:
#   1. 移除 detectron2 依賴：@configurable / TRANSFORMER_DECODER_REGISTRY /
#      Conv2d wrapper → 純 PyTorch；num_feature_levels 固定 3。
#   2. num_queries = num_classes = 19（上游 100）；query↔類別硬對應。
#   3. query_feat 初始化加 CLIP text embedding（OV-DETR 式條件化，
#      Zang et al., ECCV 2022）：q = query_feat.weight + text_feat。
#   4. 插入 1 個 condition token 進 query 序列（OneFormer task-token 的
#      插 token 變體，Jain et al., CVPR 2023）：只參與 self-attention 與 FFN，
#      不做 cross-attention、不進 prediction heads。query_embed 多配 1 個
#      PE slot 給它。
#   5. forward 簽名改為 (feats, mask_features, text_feat, cond_token)：
#      上游從 pixel decoder 取 x 與 mask_features，本專案由 SimpleFPN 提供。
#   6. 移除 self.mask_classification 分支判斷（本專案恆為 True）。
# ============================================================================
import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


# ── 以下：自上游 position_encoding.py 原樣移植（無改動） ──
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


# ── 以下四段：自上游 mask2former_transformer_decoder.py 原樣移植（僅去 detectron2 import） ──
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class M2FDecoder(nn.Module):
    """適配版 MultiScaleMaskedTransformerDecoder（適配點見檔頭清單）。"""

    def __init__(self, num_classes=19, hidden_dim=256, nheads=8,
                 dim_feedforward=2048, dec_layers=9, mask_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_feature_levels = 3  # [WeatherSAM adaptation 1] 固定 3 尺度，不走 detectron2 config

        # 上游同名成員（transformer_self_attention_layers 等三組 ModuleList）
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0,
                                   normalize_before=False))
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0,
                                    normalize_before=False))
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward,
                         dropout=0.0, normalize_before=False))

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self.query_feat = nn.Embedding(num_classes, hidden_dim)  # [WeatherSAM adaptation 2] num_queries = num_classes
        # [WeatherSAM adaptation 4] +1 slot = condition token 的 PE
        self.query_embed = nn.Embedding(num_classes + 1, hidden_dim)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 = no-object
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # ── 上游 forward_prediction_heads 原樣，唯一 adaptation：先切掉 condition token ──
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)               # (B, Q, C)
        decoder_output = decoder_output[:, : self.num_classes]        # [WeatherSAM adaptation 4]
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size,
                                  mode="bilinear", align_corners=False)
        # 上游逐字保留：
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1)
                     .repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        return outputs_class, outputs_mask, attn_mask

    def forward(self, feats, mask_features, text_feat=None, cond_token=None):
        # [WeatherSAM adaptation 5] 簽名改為 (feats, mask_features, text_feat, cond_token)
        # ── 主體對齊上游 forward；無 self.mask_classification 分支（[WeatherSAM adaptation 6]，本專案恆為 True）──
        B = mask_features.shape[0]
        src, pos, size_list = [], [], []
        for i in range(self.num_feature_levels):
            size_list.append(feats[i].shape[-2:])
            pos.append(self.pe_layer(feats[i], None).flatten(2))
            src.append(feats[i].flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)      # (HW, B, C) — 上游 seq-first 慣例
            src[-1] = src[-1].permute(2, 0, 1)

        query_embed = self.query_embed.weight[: self.num_classes]
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)        # (19, B, C)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)  # (19, B, C)
        if text_feat is not None:
            # [WeatherSAM adaptation 3] OV-DETR 式：text embedding 條件化 query
            output = output + text_feat.unsqueeze(1)
        n_cond = 0
        if cond_token is not None:
            # [WeatherSAM adaptation 4] condition token 附加於序列尾端
            output = torch.cat([output, cond_token.permute(1, 0, 2)], dim=0)  # (20, B, C)
            cond_pe = self.query_embed.weight[self.num_classes:].unsqueeze(1).repeat(1, B, 1)
            query_embed = torch.cat([query_embed, cond_pe], dim=0)
            n_cond = 1

        predictions_class, predictions_mask = [], []
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # 上游逐字保留（全空 mask fallback 全開）：
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # [WeatherSAM adaptation 4] cross-attn 只作用於前 19 個 class query
            tgt = self.transformer_cross_attention_layers[i](
                output[: self.num_classes], src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed[: self.num_classes],
            )
            output = torch.cat([tgt, output[self.num_classes:]], dim=0) if n_cond else tgt
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # 上游輸出契約：最終層 + aux（_set_aux_loss 的展開版）
        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(predictions_class[:-1], predictions_mask[:-1])
            ],
        }
        return out
