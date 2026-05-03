
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .weather_prompt_encoder import WeatherPromptEncoder
from .weather_mask_decoder import MaskDecoder
from .fusion import CMAAlignment, FlowGuidedSemanticAlignment
from .common import LayerNorm2d
from .text_encoder import TextEncoder
from .fusion_head import ResidualDWConvFusion

class WeatherSAM(nn.Module):
    """
    基於 Segment Anything Model (SAM) 改良的天氣語意分割模型 (WeatherSAM)。
    整合了 Image Encoder, Prompt Encoder, Mask Decoder，並額外加入了：
    1. CMAAlignment：UAWarpC 稠密匹配對齊晴天參考特徵，以 UAWarpC confidence 直接加權融合。
    2. ResidualDWConvFusion：Mask2Former-style 後置精修，補足 pixel 空間的局部空間一致性。
    """
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: WeatherPromptEncoder,
        mask_decoder: MaskDecoder,
        fusion_module: CMAAlignment,
        gated_fusion: FlowGuidedSemanticAlignment,
        text_encoder: TextEncoder,
        num_classes: int = 19,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.fusion_module = fusion_module
        self.gated_fusion = gated_fusion
        self.text_encoder = text_encoder

        # ConditionEncoder：天氣條件 Embedding（fog=0, rain=1, snow=2）
        self.condition_encoder = nn.Embedding(3, 256)
        nn.init.normal_(self.condition_encoder.weight, std=0.01)

        # Cityscapes 19 類的 class name → id 對照表（模型內部使用，不是可學習參數）
        self.CLASS_MAP = {
            "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
            "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
            "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
            "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
        }

        # New Semantic Fusion Head for dynamic classes
        self.num_classes = num_classes
        # [Mask2Former-style] 輕量殘差精修：3×3 DW Conv + 1×1 PW Conv + residual
        self.context_fusion_head = ResidualDWConvFusion(num_classes=num_classes)
        
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
    ) -> List[Dict[str, torch.Tensor]]:
        """
        模型的前向傳播 (Forward Pass)。

        Inputs:
            batched_input: (List[Dict]) 包含影像、文字提示、參考座標等輸入資訊。
        Returns:
            List[Dict]: 每個 Dict 包含該影像預測的 `masks`, `low_res_logits`, `class_ids`。
        """
        outputs = []
        
        # --- 階段 1：核心特徵萃取 (Image Encoding) ---
        # 1. 影像編碼 (Image Encoding) -> (B_img, 256, 64, 64)
        if "image_embedding" in batched_input[0]:
            image_embeddings = torch.stack([x["image_embedding"] for x in batched_input], dim=0)
        else:
            input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
            image_embeddings = self.image_encoder(input_images)

        # --- 階段 2：參考畫面特徵對齊與融合 (Reference Frame Alignment & Fusion) ---

        # [image-pair] 使用 clear-weather ViT-H embedding 作為 f_ref
        # 與 f_curr (foggy ViT-H embedding) 在同一特徵空間，cross-attention 的幾何對應有意義
        # clear_embedding shape: (B, 256, 64, 64)，由 DataLoader 從預算 .pt 檔案載入
        ref_embeddings = torch.stack([x["clear_embedding"] for x in batched_input], dim=0)
        ref_embeddings = ref_embeddings.to(image_embeddings.device)

        # 2. 融合 (Fusion) -> (B_img, 256, 64, 64)
        # 若 batched_input 含原始影像（cache 模式補充後），傳入 CMAAlignment 供 VGG 使用
        img_curr_list = [x.get('image') for x in batched_input]
        img_ref_list  = [x.get('clear_image') for x in batched_input]
        if all(t is not None for t in img_curr_list):
            img_curr_batch = torch.stack(img_curr_list, dim=0)
            img_ref_batch  = torch.stack(img_ref_list,  dim=0)
        else:
            img_curr_batch = img_ref_batch = None

        f_ref_warped, _ = self.fusion_module(
            f_curr=image_embeddings,
            f_ref=ref_embeddings,
            img_curr=img_curr_batch,
            img_ref=img_ref_batch,
        )
        # CMAAlignment：VGG flow → warp(f_ref) → hard confidence mask → f_ref_warped
        # GatedFusion：f_curr + alpha * f_ref_warped（加法注入，f_curr 完整保留）
        fused_embeddings = self.gated_fusion(image_embeddings, f_ref_warped)

        # --- 階段 3：批次提示編碼與遮罩解碼 (Prompt Encoding & Mask Decoding) ---
        for i, image_record in enumerate(batched_input):
            # A. 文字編碼 (Text Encoding)
            texts = image_record["text_prompts"] # List[str], len = K
            
            # 取得原始 Embeddings
            sparse_embeddings = self.text_encoder(texts) 
            
            # 維度防呆 (確保是 K, 1, 256)
            if sparse_embeddings.dim() == 2: # (K, 256)
                sparse_embeddings = sparse_embeddings.unsqueeze(1)
            elif sparse_embeddings.dim() == 3 and sparse_embeddings.shape[0] == 1: # (1, K, 256)
                sparse_embeddings = sparse_embeddings.squeeze(0).unsqueeze(1)

            # B. 條件編碼（固定使用 ConditionEncoder：fog=0, rain=1, snow=2）
            condition_id = image_record.get("condition_id", None)
            if condition_id is None:
                import warnings
                warnings.warn("condition_id not found in input, defaulting to 0 (fog). "
                              "Ensure ACDC CSV has condition_id column.", stacklevel=2)
                condition_id = torch.tensor(0)
            cid = condition_id.to(self.device).unsqueeze(0)  # (1,)
            loc_feats = self.condition_encoder(cid)           # (1, 256)
            loc_feats = F.normalize(loc_feats, p=2, dim=-1)
            
            # 3. 調整維度以匹配 Text Prompts
            # 目標: (K, 1, 256) 以便與 (K, 1, 256) 的 Text 串接
            k_prompts = sparse_embeddings.shape[0]
            
            # (1, 256) -> (1, 1, 256)
            loc_feats = loc_feats.unsqueeze(1)
            # (1, 1, 256) -> (K, 1, 256) 複製 K 份
            location_embeddings = loc_feats.repeat(k_prompts, 1, 1)

            
            # C. 提示編碼 (Prompt Encoding) → sparse: (K, 2, 256)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                text_embeddings=sparse_embeddings,
                location_embeddings=location_embeddings,
            )

            # D. [Mask2Former-style] 語意解碼 — 所有 K 個類別在單次 transformer forward 中處理
            # class_ids 必須與 texts 完全同步（同長度、同順序），確保 sparse_embeddings 對應正確
            # 若 texts 中有不在 CLASS_MAP 的字串則過濾，同時從 sparse_embeddings 中移除對應項
            valid_indices = [k for k, t in enumerate(texts) if t in self.CLASS_MAP]
            class_ids = [self.CLASS_MAP[texts[k]] for k in valid_indices]
            if len(valid_indices) < len(texts):
                sparse_embeddings = sparse_embeddings[valid_indices]  # 同步過濾 embeddings

            if not class_ids:
                h, w = image_record["original_size"]
                outputs.append({
                    "masks": torch.zeros(1, 0, h, w, device=self.device),
                    "low_res_logits": torch.zeros(1, 0, 256, 256, device=self.device),
                    "class_ids": [],
                })
                continue

            curr_fused_embed = fused_embeddings[i].unsqueeze(0)  # (1, 256, 64, 64)
            image_pe = self.get_image_pe()

            # forward_semantic：K 個類別 query 在同一 sequence 中互相 attend
            # 輸出 low_res_masks: (1, K, 256, 256)
            low_res_masks = self.mask_decoder.forward_semantic(
                image_embeddings=curr_fused_embed,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,   # (K, 2, 256)
                dense_prompt_embeddings=dense_embeddings,
                class_ids=class_ids,
            )

            # E. Post-processing: (1, K, 256, 256) → (1, K, H, W)
            # context_fusion_head 在 trainer Stage 5 呼叫（含完整 loss pipeline），
            # 此處不重複應用，直接將 raw decoder 輸出送入 postprocess。
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["original_size"],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "masks": masks,                  # (1, K, H, W)
                    "low_res_logits": low_res_masks, # (1, K, 256, 256)
                    "class_ids": class_ids,          # List[int], len=K
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