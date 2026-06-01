# build_weather_sam.py
import torch
from functools import partial

from .modeling import ImageEncoderViT, TwoWayTransformer, WeatherPromptEncoder, CMAAlignment, TextEncoder, WeatherSAM, MaskDecoder

def build_weather_sam_vit_b(num_classes=19, checkpoint=None):
    return _build_weather_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )

def build_weather_sam_vit_h(num_classes=19, checkpoint=None):
    return _build_weather_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )

def build_weather_sam_from_config(cfg: dict, checkpoint=None):
    """[ablation] 依 config dict 建構 WeatherSAM，統一 train 與 eval 的建模路徑。

    cfg keys: model_type, use_vgg_adapter(bool), inject('pre'/'post'),
              decoder('unified'/'per_class'), lrh(bool), mfb(bool), ref(bool)
    注意：mfb 屬 loss 端（在 trainer 設定），不在模型建構處理。
    """
    if cfg.get('model_type', 'vit_h') == 'vit_b':
        model = build_weather_sam_vit_b(checkpoint=checkpoint)
    else:
        model = build_weather_sam_vit_h(checkpoint=checkpoint)

    model.use_lrh = bool(cfg.get('lrh', True))
    model.mask_decoder.decoder_mode = cfg.get('decoder', 'unified')
    model.vgg_injector.use_reference = bool(cfg.get('ref', True))

    if cfg.get('use_vgg_adapter', True):
        model.enable_vgg_adapter(mode=cfg.get('inject', 'pre'))

    return model


weather_sam_model_registry = {
    "default": build_weather_sam_vit_h,
    "vit_h": build_weather_sam_vit_h,
    "vit_b": build_weather_sam_vit_b,
}

def _build_weather_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes=19,
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

    prompt_encoder = WeatherPromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
    )

    mask_decoder = MaskDecoder(
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        num_classes=num_classes,
    )

    fusion_module = CMAAlignment(
        embed_dim=prompt_embed_dim,
        pretrained_path="/home/rvl1421/SAM_research-1/segment-anything/checkpoints/cma_alignment_weights.pth",
        confidence_threshold=0.2,
    )

    text_encoder = TextEncoder(
        model_name="ViT-B/32", # CLIP model
        output_dim=prompt_embed_dim,
        freeze=True
    )

    # 2. 組合 WeatherSAM
    sam = WeatherSAM(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        fusion_module=fusion_module,
        text_encoder=text_encoder,
        num_classes=num_classes,
    )

    # 3. 載入預訓練權重 (如果有提供)
    if checkpoint is not None:
        print(f"Loading weights from {checkpoint}...")
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, weights_only=False)
            
        # 🌟 【關鍵修復】檢查這是否是由 Trainer 儲存的 checkpoint 字典
        if 'model_state_dict' in state_dict:
            print("Detected Trainer checkpoint. Extracting 'model_state_dict'...")
            state_dict = state_dict['model_state_dict']
            
        # 過濾不匹配的鍵值
        model_dict = sam.state_dict()
        
        # 僅載入匹配的 Image Encoder 與部分 Decoder 權重
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        missing_keys = [k for k in model_dict if k not in pretrained_dict]
        print(f"Missing keys (New modules initialized from scratch): {len(missing_keys)}")

        # 👇 加入這兩行來進行診斷
        if len(missing_keys) > 0:
            print("🕵️ 遺失的前 10 個權重名稱:")
            for mk in missing_keys[:10]:
                print(f"   - {mk}")
        
        model_dict.update(pretrained_dict)
        sam.load_state_dict(model_dict)
        
    return sam