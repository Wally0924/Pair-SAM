# build_weather_sam.py
import torch
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, TwoWayTransformer, MaskEncoder, WeatherPromptEncoder, CrossViewAlignment, GatedFusion, TextEncoder, WeatherSAM

def build_weather_sam_vit_b(checkpoint=None):
    return _build_weather_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_weather_sam_vit_h(checkpoint=None):
    return _build_weather_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

def _build_weather_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # 1. 實例化各個組件
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    mask_encoder = MaskEncoder(
        in_chans=3, 
        embed_dim=prompt_embed_dim
    )

    prompt_encoder = WeatherPromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    fusion_module = CrossViewAlignment(
        embed_dim=prompt_embed_dim,
        num_heads=8
    )

    gate_module = GatedFusion(
        embed_dim=prompt_embed_dim
    )
    
    text_encoder = TextEncoder(
        model_name="ViT-B/32", # CLIP model
        output_dim=prompt_embed_dim,
        freeze=True
    )

    # 2. 組合 WeatherSAM
    sam = WeatherSAM(
        image_encoder=image_encoder,
        mask_encoder=mask_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        fusion_module=fusion_module,
        gate_module=gate_module,
        text_encoder=text_encoder,
    )

    # 3. 載入預訓練權重 (如果有提供)
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        
        # 過濾掉不匹配的鍵值 (因為我們有新模組)
        # 這裡我們只載入 image_encoder 和 mask_decoder 的權重
        # 注意: 如果你是從原始 SAM checkpoint 載入，key 可能需要調整 (例如加前綴 'image_encoder.')
        # 這裡假設 checkpoint 是標準 SAM 格式
        # [新增邏輯] 檢查是否為新格式 (包含 config 的 dict)
        if "model_state_dict" in state_dict:
            print("📦 Detected new checkpoint format (with config).")
            # 如果你有需要讀 config，可以在這裡讀 state_dict['config']
            state_dict = state_dict["model_state_dict"] # 取出真正的權重
        
        sam_dict = sam.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in sam_dict and v.shape == sam_dict[k].shape}
        
        # 載入匹配的權重 (ViT, Decoder)
        sam.load_state_dict(pretrained_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} keys from checkpoint.")
        
    return sam