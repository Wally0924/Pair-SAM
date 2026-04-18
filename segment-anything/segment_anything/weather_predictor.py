import torch
import torch.nn.functional as F

class WeatherSamPredictor:
    def __init__(self, sam_model) -> None:
        """
        將 WeatherSAM 封裝為標準的 Predictor 介面。
        確保高成本的影像/遮罩融合特徵只計算一次。
        """
        self.model = sam_model
        self.device = sam_model.device
        self.reset_image()

    def reset_image(self) -> None:
        """重置當前狀態"""
        self.is_image_set = False
        self.fused_features = None
        self.original_size = None

    @torch.no_grad()
    def set_image_data(
        self, 
        image: torch.Tensor = None, 
        image_embedding: torch.Tensor = None, 
        reference_mask: torch.Tensor = None, 
        ref_void_mask: torch.Tensor = None, 
        original_size: tuple = (1024, 1024)
    ) -> None:
        """
        處理影像與 Reference Mask，並快取 Fusion 後的特徵。
        """
        self.reset_image()
        self.original_size = original_size
        
        # 1. 取得影像特徵 (支援直接傳入預先算好的 Embedding 以加速)
        if image_embedding is not None:
            img_embed = image_embedding.unsqueeze(0) # (1, 256, 64, 64)
        else:
            if image is None:
                raise ValueError("Must provide either 'image' or 'image_embedding'.")
            # 這裡的 image 必須是 (3, 1024, 1024) 且未經正規化的 Tensor
            input_image = self.model.preprocess(image).unsqueeze(0)
            img_embed = self.model.image_encoder(input_image)

        # 2. 處理 Reference Mask
        ref_mask = reference_mask.float() / 255.0
        h, w = ref_mask.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        ref_mask = F.pad(ref_mask, (0, padw, 0, padh)).unsqueeze(0)
        
        ref_embed = self.model.mask_encoder(ref_mask)
        ref_void = ref_void_mask.unsqueeze(0)

        # 3. 執行特徵融合 (CrossViewAlignment & GatedFusion)
        aligned_embed = self.model.fusion_module(f_curr=img_embed, f_ref=ref_embed)
        self.fused_features = self.model.gate_module(f_curr=img_embed, f_align=aligned_embed)
        
        self.is_image_set = True

    @torch.no_grad()
    def predict(
        self, 
        text_prompts: list, 
        location: torch.Tensor, 
        multimask_output: bool = True
    ):
        """
        給定文本與地理位置，使用已快取的融合特徵預測遮罩。
        """
        if not self.is_image_set:
            raise RuntimeError("Must set image data with .set_image_data(...) before predicting.")

        # --- A. Text Encoding ---
        sparse_embeddings = self.model.text_encoder(text_prompts) 
        if sparse_embeddings.dim() == 2:
            sparse_embeddings = sparse_embeddings.unsqueeze(1)
        elif sparse_embeddings.dim() == 3 and sparse_embeddings.shape[0] == 1:
            sparse_embeddings = sparse_embeddings.squeeze(0).unsqueeze(1)

        # --- B. Location / Condition Encoding ---
        condition_id = self.model.CLASS_MAP.get('__condition__', None)  # 由外部透過 set_condition 設定
        if hasattr(self, '_condition_id') and self._condition_id >= 0:
            cid = torch.tensor(self._condition_id, dtype=torch.long, device=self.device).unsqueeze(0)
            loc_feats = self.model.condition_encoder(cid)
        else:
            location_coords = location.unsqueeze(0)
            loc_feats = self.model.location_encoder(location_coords)
        loc_feats = F.normalize(loc_feats, p=2, dim=-1)

        k_prompts = sparse_embeddings.shape[0]
        loc_feats = loc_feats.unsqueeze(1)
        location_embeddings = loc_feats.repeat(k_prompts, 1, 1)

        # --- C. Prompt Fusion ---
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            text_embeddings=sparse_embeddings,
            location_embeddings=location_embeddings,
        )

        image_pe = self.model.get_image_pe()

        # --- D. [Mask2Former] 語意解碼 ---
        class_ids = [self.model.CLASS_MAP[t] for t in text_prompts if t in self.model.CLASS_MAP]
        low_res_masks = self.model.mask_decoder.forward_semantic(
            image_embeddings=self.fused_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,   # (K, 2, 256)
            dense_prompt_embeddings=dense_embeddings,
            class_ids=class_ids,
        )
        # low_res_masks: (1, K, 256, 256)

        # --- E. 後處理 ---
        masks = self.model.postprocess_masks(
            low_res_masks,
            input_size=(1024, 1024),
            original_size=self.original_size,
        )
        # 回傳 logits，以利後續做 Argmax；同時回傳 class_ids 供 inference 組裝
        return masks, low_res_masks, class_ids

    def set_condition(self, condition_id: int):
        """設定天氣條件 ID（ACDC 模式：fog=0, rain=1, snow=2；-1 = GPS 模式）"""
        self._condition_id = condition_id