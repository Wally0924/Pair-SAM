# weather_sam.py
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

# 引入所有定義好的模組
from .image_encoder import ImageEncoderViT
from .mask_encoder import MaskEncoder
from .weather_prompt_encoder import WeatherPromptEncoder
from .weather_mask_decoder import MaskDecoder
from .fusion import CrossViewAlignment, GatedFusion
from .text_encoder import TextEncoder

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
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        WeatherSAM: 抗惡劣天氣的語意分割模型 (Retrieval-Augmented SAM)
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.fusion_module = fusion_module
        self.gate_module = gate_module
        self.text_encoder = text_encoder
        
        # 影像預處理參數
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # 補充位置編碼 (Positional Encoding) - 這是 Decoder 必需的
        # 這裡使用 1x256x64x64 的可學習參數，與 SAM 預設一致
        embed_dim = prompt_encoder.embed_dim
        image_embedding_size = prompt_encoder.image_embedding_size
        self.pe_layer = nn.Parameter(torch.zeros(1, embed_dim, *image_embedding_size))
        nn.init.normal_(self.pe_layer, std=0.02) # 初始化

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def get_image_pe(self) -> torch.Tensor:
        """回傳位置編碼 (1, 256, 64, 64)"""
        return self.pe_layer

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward Pass
        """
        outputs = []
        
        # ------------------------------------------------------------------
        # 1. Image Encoding (ViT) -> F_curr
        # ------------------------------------------------------------------
        # [修改 1]：支援 Cached Features (加速訓練用)
        if "image_embedding" in batched_input[0]:
            image_embeddings = torch.stack([x["image_embedding"] for x in batched_input], dim=0)
        else:
            input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
            image_embeddings = self.image_encoder(input_images) # (B, 256, 64, 64)

        # ------------------------------------------------------------------
        # 2. Mask Encoding (Reference Mask) -> F_ref
        # ------------------------------------------------------------------
        # [修改 2]：針對 MaskEncoder (自定義CNN) 改用 0~1 標準化
        ref_masks_list = []
        for x in batched_input:
            mask = x["reference_mask"] # (3, H, W) 來自 DataLoader,數值 0-255
            
            # A. 轉為 0~1 float
            mask = mask.float() / 255.0
            
            # B. Padding (確保尺寸對齊 1024，雖然 DataLoader 通常已 Resize，但保留以防萬一)
            h, w = mask.shape[-2:]
            padh = self.image_encoder.img_size - h
            padw = self.image_encoder.img_size - w
            # F.pad 參數順序: (left, right, top, bottom)
            mask = F.pad(mask, (0, padw, 0, padh))
            
            ref_masks_list.append(mask)

        # 堆疊成 Batch Tensor: (B, 3, 1024, 1024)
        ref_masks = torch.stack(ref_masks_list, dim=0)
        
        # 進入 Mask Encoder
        ref_embeddings = self.mask_encoder(ref_masks) # (B, 256, 64, 64)

        # ------------------------------------------------------------------
        # 3. Fusion (Alignment & Gating) -> F_fuse
        # ------------------------------------------------------------------
        aligned_embeddings = self.fusion_module(f_curr=image_embeddings, f_ref=ref_embeddings)
        fused_embeddings = self.gate_module(f_curr=image_embeddings, f_align=aligned_embeddings)

        # ------------------------------------------------------------------
        # 4. Prompt Encoding & Decoding
        # ------------------------------------------------------------------
        for i, image_record in enumerate(batched_input):
            # A. Text Encoding
            texts = image_record["text_prompts"]
            sparse_embeddings = self.text_encoder(texts)
            if sparse_embeddings.shape[0] == 1 and sparse_embeddings.shape[1] > 1:
                sparse_embeddings = sparse_embeddings.permute(1, 0, 2)

            # B. Weather Prompt Encoding
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                text_embeddings=sparse_embeddings,
                mask_inputs=None 
            )

            # C. Mask Decoding
            curr_fused_embed = fused_embeddings[i].unsqueeze(0)
            image_pe = self.get_image_pe()

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
                input_size=image_record["original_size"], # 這裡注意 key 要對應 DataLoader
                original_size=image_record["original_size"],
            )
            
            # Binary Mask Threshold
            masks = masks > self.mask_threshold
            
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
            
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """將 256x256 的遮罩還原回原始影像尺寸"""
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
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x