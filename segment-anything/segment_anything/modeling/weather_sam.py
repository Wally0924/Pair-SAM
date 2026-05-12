
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .weather_prompt_encoder import WeatherPromptEncoder
from .weather_mask_decoder import MaskDecoder
from .fusion import CMAAlignment
from .text_encoder import TextEncoder
from .fusion_head import ResidualDWConvFusion
from .vgg_adapter import MultiScaleCrossAttnInjector

class WeatherSAM(nn.Module):
    """
    基於 Segment Anything Model (SAM) 改良的天氣語意分割模型 (WeatherSAM)。
    整合了 Image Encoder, Prompt Encoder, Mask Decoder，並額外加入了：
    1. CMAAlignment：UAWarpC 稠密匹配對齊晴天參考特徵，輸出 f_ref_warped + confidence map
       （目前作為診斷監控與 Phase-2 預備，不再注入 decoder）。
    2. MultiScaleCrossAttnInjector：將 clear-ref VGG l2/l3 經 UAWarpC 對齊、confidence < 0.2
       位置歸零後，透過瓶頸式 Cross-Attention 注入 ViT-H Encoder Block 7/15/23/31 的 token。
    3. ResidualDWConvFusion：Mask2Former-style 後置精修。
    """
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: WeatherPromptEncoder,
        mask_decoder: MaskDecoder,
        fusion_module: CMAAlignment,
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
        self.text_encoder = text_encoder

        # ConditionEncoder：天氣條件 Embedding（fog=0, rain=1, snow=2, night=3）
        self.condition_encoder = nn.Embedding(4, 256)
        nn.init.normal_(self.condition_encoder.weight, std=0.01)

        # Cityscapes 19 類的 class name → id 對照表（模型內部使用，不是可學習參數）
        self.CLASS_MAP = {
            "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
            "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
            "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
            "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
        }

        self.num_classes = num_classes
        # [Mask2Former-style] 輕量殘差精修：3×3 DW Conv + 1×1 PW Conv + residual
        self.context_fusion_head = ResidualDWConvFusion(num_classes=num_classes)

        # [testv14] Dense Prompt Fusion 路徑已移除：跨條件補償改由 WarpedVGG Adapter
        # 在 ViT Encoder 內部完成，避免 encoder-level 與 decoder-level 雙重注入造成
        # gate 互相競爭、稀釋 adapter 學習信號。Decoder 的 dense prompt 改用
        # WeatherPromptEncoder 內建的 no_mask_embed（中性零訊號）。

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        embed_dim = prompt_encoder.embed_dim
        image_embedding_size = prompt_encoder.image_embedding_size
        self.pe_layer = nn.Parameter(torch.zeros(1, embed_dim, *image_embedding_size))
        nn.init.normal_(self.pe_layer, std=0.02)

        # [multi-stage] WarpedVGG Adapter：在 ViT-H Block 7/15/23/31（4 個 global attention block）
        # 分段注入對齊後的晴天參考 VGG 特徵，讓 early/mid/late 各層都能感知補償信號。
        # use_vgg_adapter=False 時完全跳過，hook 未註冊，不影響現有 forward 行為。
        self.use_vgg_adapter: bool = False
        self.vgg_injector = MultiScaleCrossAttnInjector(
            vit_dim=1280, d_attn=256, l2_channels=256, l3_channels=512,
            d_kv=64, pool_size=32, num_heads=4, gate_init=-5.0,
        )
        self._adapter_hook_handles: list = []

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def get_image_pe(self) -> torch.Tensor:
        return self.pe_layer

    def enable_vgg_adapter(self, mode: str = 'pre'):
        """啟用 MultiScaleCrossAttnInjector，在 ViT-H Block [7, 15, 23, 31] 各注冊一個 hook。

        Args:
            mode: 'pre'（預設）在 block 自注意力前注入，補償信號參與 Q/K/V 計算；
                  'post' 在 block 自注意力後注入，保留原 post-hook 行為供消融實驗使用。

        4 個注入點對應 ViT-H 的 global attention blocks，使 encoder early/mid/late
        各段都能感知對齊後的晴天參考特徵。每次呼叫前先移除舊 hook 避免重複注入。
        """
        if mode not in ('pre', 'post'):
            raise ValueError(f"[WeatherSAM] mode must be 'pre' or 'post', got {mode!r}")

        for handle in self._adapter_hook_handles:
            handle.remove()
        self._adapter_hook_handles = []

        all_inject_blocks = self.vgg_injector.INJECT_BLOCKS
        n_blocks = len(self.image_encoder.blocks)
        inject_blocks = [b for b in all_inject_blocks if b < n_blocks]
        if len(inject_blocks) != len(all_inject_blocks):
            warnings.warn(
                f"[WeatherSAM] Some INJECT_BLOCKS out of range for {n_blocks}-block encoder; "
                f"using {inject_blocks} instead of {all_inject_blocks}.",
                stacklevel=2,
            )
        for stage_idx, block_idx in enumerate(inject_blocks):
            target_block = self.image_encoder.blocks[block_idx]
            if mode == 'pre':
                handle = target_block.register_forward_pre_hook(
                    self.vgg_injector._make_pre_hook(stage_idx)
                )
            else:
                handle = target_block.register_forward_hook(
                    self.vgg_injector._make_hook(stage_idx)
                )
            self._adapter_hook_handles.append(handle)

        self.use_vgg_adapter = True
        print(f'[WeatherSAM] MultiStage WarpedVGG Adapter enabled at ViT Blocks {inject_blocks} (mode={mode!r}).')

    def disable_vgg_adapter(self):
        """停用 MultiScale WarpedVGG Adapter，移除所有已注冊的 hook。"""
        for handle in self._adapter_hook_handles:
            handle.remove()
        self._adapter_hook_handles = []
        self.use_vgg_adapter = False

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

        # --- 階段 0：pre_align — UAWarpC 對齊後的 VGG ref 特徵供 WarpedVGG Adapter 使用 ---
        # CMAAlignment.pre_align() 僅使用原始影像的 VGG 特徵計算 flow，不依賴 ViT embedding，
        # 因此可在 SAM Encoder 之前安全執行，不存在循環依賴。
        _vgg_ref_aligned = None
        if (self.use_vgg_adapter
                and "image_embedding" not in batched_input[0]
                and all("image" in x for x in batched_input)
                and all("clear_image" in x for x in batched_input)):
            img_curr_batch = torch.stack(
                [x["image"] for x in batched_input], dim=0
            ).to(self.device)
            img_ref_batch = torch.stack(
                [x["clear_image"] for x in batched_input], dim=0
            ).to(self.device)
            _vgg_ref_aligned = self.fusion_module.pre_align(img_curr_batch, img_ref_batch)
            # pre_align 在 no_grad 下會產生大量 VGG 中間特徵（1024×1024 雙圖）
            # 釋放 CUDA allocator cache，為後續 ViT-H global attention (1 GiB) 騰出空間
            torch.cuda.empty_cache()

        # --- 階段 1：核心特徵萃取 (Image Encoding) ---
        if "image_embedding" in batched_input[0]:
            image_embeddings = torch.stack([x["image_embedding"] for x in batched_input], dim=0)
        else:
            input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
            if self.use_vgg_adapter and _vgg_ref_aligned is not None:
                self.vgg_injector.set_features(_vgg_ref_aligned)
            image_embeddings = self.image_encoder(input_images)

        # --- 階段 2：批次提示編碼與遮罩解碼 (Prompt Encoding & Mask Decoding) ---
        for i, image_record in enumerate(batched_input):
            # A. 文字編碼 (Text Encoding)
            texts = image_record["text_prompts"]

            sparse_embeddings = self.text_encoder(texts)

            if sparse_embeddings.dim() == 2:
                sparse_embeddings = sparse_embeddings.unsqueeze(1)
            elif sparse_embeddings.dim() == 3 and sparse_embeddings.shape[0] == 1:
                sparse_embeddings = sparse_embeddings.squeeze(0).unsqueeze(1)

            # B. 條件編碼（fog=0, rain=1, snow=2, night=3）
            condition_id = image_record.get("condition_id", None)
            if condition_id is None:
                warnings.warn("condition_id not found in input, defaulting to 0 (fog). "
                              "Ensure ACDC CSV has condition_id column.", stacklevel=2)
                condition_id = torch.tensor(0)
            cid = condition_id.to(self.device).unsqueeze(0)
            loc_feats = self.condition_encoder(cid)
            loc_feats = F.normalize(loc_feats, p=2, dim=-1)

            k_prompts = sparse_embeddings.shape[0]
            loc_feats = loc_feats.unsqueeze(1)
            location_embeddings = loc_feats.repeat(k_prompts, 1, 1)

            # C. 提示編碼 (Prompt Encoding) → sparse: (K, 2, 256)
            # [testv14] dense_embeddings 改用 WeatherPromptEncoder 的 no_mask_embed（中性零訊號），
            # 跨條件補償已由 WarpedVGG Adapter 在 encoder 內完成，無需 decoder 入口再注入
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                text_embeddings=sparse_embeddings,
                location_embeddings=location_embeddings,
            )

            # D. 過濾有效類別
            valid_indices = [k for k, t in enumerate(texts) if t in self.CLASS_MAP]
            class_ids = [self.CLASS_MAP[texts[k]] for k in valid_indices]
            if len(valid_indices) < len(texts):
                sparse_embeddings = sparse_embeddings[valid_indices]

            if not class_ids:
                h, w = image_record["original_size"]
                outputs.append({
                    "masks": torch.zeros(1, 0, h, w, device=self.device),
                    "low_res_logits": torch.zeros(1, 0, 256, 256, device=self.device),
                    "class_ids": [],
                })
                continue

            image_pe = self.get_image_pe()

            # E. [Mask2Former-style] 語意解碼
            # image_embeddings[i]：f_curr（已含 UAWarpC 對齊的 WarpedVGG 注入補償）
            # dense_prompt_embeddings：no_mask_embed 中性訊號
            low_res_masks = self.mask_decoder.forward_semantic(
                image_embeddings=image_embeddings[i].unsqueeze(0),  # (1, 256, 64, 64)
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,          # (K, 2, 256)
                dense_prompt_embeddings=dense_embeddings[:1],        # (1, 256, 64, 64)
                class_ids=class_ids,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["original_size"],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "masks": masks,
                    "low_res_logits": low_res_masks,
                    "class_ids": class_ids,
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
