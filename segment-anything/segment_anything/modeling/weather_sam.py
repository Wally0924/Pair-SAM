
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_encoder import MaskEncoder
from .weather_prompt_encoder import WeatherPromptEncoder
from .weather_mask_decoder import MaskDecoder
from .fusion import CrossViewAlignment, GatedFusion
from .text_encoder import TextEncoder
from .location_encoder import LocationEncoder
from .fusion_head import SemanticFusionHead

class WeatherSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        mask_encoder: MaskEncoder,
        prompt_encoder: WeatherPromptEncoder,
        mask_decoder: MaskDecoder,
        fusion_module: CrossViewAlignment,
        gate_module: GatedFusion,
        text_encoder: TextEncoder,
        location_encoder: LocationEncoder,
        num_classes: int = 19,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.fusion_module = fusion_module
        self.gate_module = gate_module
        self.text_encoder = text_encoder
        self.location_encoder = location_encoder
        
        # New Semantic Fusion Head for dynamic classes
        self.num_classes = num_classes
        self.semantic_fusion_head = SemanticFusionHead(num_classes=num_classes, hidden_dim=64)
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        embed_dim = prompt_encoder.embed_dim
        image_embedding_size = prompt_encoder.image_embedding_size
        self.pe_layer = nn.Parameter(torch.zeros(1, embed_dim, *image_embedding_size))
        nn.init.normal_(self.pe_layer, std=0.02)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def get_image_pe(self) -> torch.Tensor:
        return self.pe_layer

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        outputs = []
        
        # 1. Image Encoding -> (B_img, 256, 64, 64)
        if "image_embedding" in batched_input[0]:
            image_embeddings = torch.stack([x["image_embedding"] for x in batched_input], dim=0)
        else:
            input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
            image_embeddings = self.image_encoder(input_images)

        # 2. Mask Encoding -> (B_img, 256, 64, 64)
        ref_masks_list = []
        for x in batched_input:
            mask = x["reference_mask"] 
            mask = mask.float() / 255.0 # Normalize 0-1
            h, w = mask.shape[-2:]
            padh = self.image_encoder.img_size - h
            padw = self.image_encoder.img_size - w
            mask = F.pad(mask, (0, padw, 0, padh))
            ref_masks_list.append(mask)

        ref_masks = torch.stack(ref_masks_list, dim=0)
        ref_embeddings = self.mask_encoder(ref_masks)
        ref_void_masks = torch.stack([x["ref_void_mask"] for x in batched_input], dim=0)

        # 3. Fusion -> (B_img, 256, 64, 64)
        aligned_embeddings = self.fusion_module(f_curr=image_embeddings, f_ref=ref_embeddings, ref_void_mask=ref_void_masks)
        fused_embeddings = self.gate_module(f_curr=image_embeddings, f_align=aligned_embeddings, ref_void_mask=ref_void_masks)

        # 4. Prompt Encoding & Decoding Loop
        for i, image_record in enumerate(batched_input):
            # A. Text Encoding
            texts = image_record["text_prompts"] # List[str], len = K
            
            # 取得原始 Embeddings
            sparse_embeddings = self.text_encoder(texts) 
            
            # 維度防呆 (確保是 K, 1, 256)
            if sparse_embeddings.dim() == 2: # (K, 256)
                sparse_embeddings = sparse_embeddings.unsqueeze(1)
            elif sparse_embeddings.dim() == 3 and sparse_embeddings.shape[0] == 1: # (1, K, 256)
                sparse_embeddings = sparse_embeddings.squeeze(0).unsqueeze(1)

            # --- [Step B] Location Encoding (Call like Text Encoder) ---
            # 1. 取出座標 (2,) -> (1, 2) Batch dimension
            location_coords = image_record["location"].unsqueeze(0) 
            
            # [cite_start]2. 編碼: (1, 2) -> (1, 256) [cite: 219]
            loc_feats = self.location_encoder(location_coords)
            loc_feats = F.normalize(loc_feats, p=2, dim=-1)
            
            # 3. 調整維度以匹配 Text Prompts
            # 目標: (K, 1, 256) 以便與 (K, 1, 256) 的 Text 串接
            k_prompts = sparse_embeddings.shape[0]
            
            # (1, 256) -> (1, 1, 256)
            loc_feats = loc_feats.unsqueeze(1)
            # (1, 1, 256) -> (K, 1, 256) 複製 K 份
            location_embeddings = loc_feats.repeat(k_prompts, 1, 1)

            
            # B. Weather Prompt Encoding (K, 1, 256)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                text_embeddings=sparse_embeddings,
                mask_inputs=None,
                location_embeddings=location_embeddings,  
            )

            # C. Mask Decoding
            # 取出當前這張圖的 Feature: (1, 256, 64, 64)
            curr_fused_embed = fused_embeddings[i].unsqueeze(0)
            image_pe = self.get_image_pe()

            # Decoder 內部會檢測:
            # Image Embed: (1, ...) 
            # Prompts: (K, ...)
            # 自動執行 repeat_interleave，將 Image 複製 K 份
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_fused_embed,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # D. Post-processing
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["original_size"],
                original_size=image_record["original_size"],
            )
            
            # E. Semantic Fusion (Optional but recommended for semantic segmentation)
            # low_res_masks is (K, 3, 256, 256). We typically take the best mask (index 0 if multimask=False, or from iou).
            # We will process it after upsampling to original size, OR at 256 resolution.
            # Doing it at high-res provides best edge quality. However, passing K back and handling later is cleaner.
            
            outputs.append(
                {
                    "masks": masks,              # (K, 3, H, W) Boolean / Float masks
                    "iou_predictions": iou_predictions, # (K, 3)
                    "low_res_logits": low_res_masks,    # (K, 3, 256, 256)
                }
            )
            
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x