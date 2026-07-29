# build_pair_sam.py
import torch
from functools import partial

from .modeling import ImageEncoderViT, TwoWayTransformer, PairPromptEncoder, CMAAlignment, TextEncoder, PairSAM, MaskDecoder

def build_pair_sam_vit_b(num_classes=19, checkpoint=None, adapter_variant='reference'):
    return _build_pair_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
        adapter_variant=adapter_variant,
    )

def build_pair_sam_vit_h(num_classes=19, checkpoint=None, adapter_variant='reference'):
    return _build_pair_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
        adapter_variant=adapter_variant,
    )

def build_pair_sam_from_config(cfg: dict, checkpoint=None):
    """[ablation] 依 config dict 建構 PairSAM，統一 train 與 eval 的建模路徑。

    cfg keys: model_type, use_vgg_adapter(bool), inject('pre'/'post'),
              decoder('unified'/'per_class'), lrh(bool), mfb(bool), ref(bool), cond(bool)
    注意：mfb 屬 loss 端（在 trainer 設定），不在模型建構處理。
    """
    # [ablation B] adapter 變體穿進 builder，使 SAM-Adapter 注入器在「載入 checkpoint 之前」
    # 就建好，否則 _build 會先以參考注入器載入、隨後被換掉而丟失已訓練的 adapter 權重
    # （eval/resume 時會載入失敗 → 注入器變隨機）。
    variant = cfg.get('adapter_variant', 'reference')
    if cfg.get('model_type', 'vit_h') == 'vit_b':
        model = build_pair_sam_vit_b(checkpoint=checkpoint, adapter_variant=variant)
    else:
        model = build_pair_sam_vit_h(checkpoint=checkpoint, adapter_variant=variant)

    model.use_lrh = bool(cfg.get('lrh', True))
    model.use_cond = bool(cfg.get('cond', True))
    _dec = cfg.get('decoder', 'unified')
    if _dec == 'm2f':
        model.decoder_arch = 'm2f'
        model.use_lrh = False  # LRH/fusion head 屬 legacy 組裝路徑，m2f 不經過
    else:
        model.mask_decoder.decoder_mode = _dec
    _use_ref = bool(cfg.get('ref', True))
    if hasattr(model.vgg_injector, 'rpm'):
        model.vgg_injector.use_reference = _use_ref
        model.vgg_injector.rpm.use_reference = _use_ref  # propagate to RPM for ablation
        # W3 消融:移除置信度調變(m̄≡1,參考特徵不分可靠與否全幅注入)
        model.vgg_injector.rpm.use_conf_mod = bool(cfg.get('conf_mod', True))
        # W6 消融:移除抽取器(單向注入,參考先驗不隨主幹深度更新)。
        # 必須在 enable_vgg_adapter(hook 註冊)之前清空:extract hook 依 EXTRACT_BLOCKS
        # 註冊,清空後不註冊、extractors 無參數 → 無 trainable-but-unused(梯度稽核不誤報)。
        # checkpoint 載入為過濾式(僅載形狀相符鍵),extractor 鍵缺席不影響 eval/resume。
        if not cfg.get('extractor', True):
            model.vgg_injector.EXTRACT_BLOCKS = []
            model.vgg_injector.extractors = torch.nn.ModuleList()
            model.vgg_injector._extract_c = []
    # else: sam_adapter 基線無 rpm 且恆 reference-free(use_reference=False 為其不變量),
    #       不以 --ref/--no-conf_mod/--no-extractor 覆蓋;此三軸僅對 reference 變體有意義

    if cfg.get('use_vgg_adapter', True):
        model.enable_vgg_adapter(mode=cfg.get('inject', 'pre'))

    return model


pair_sam_model_registry = {
    "default": build_pair_sam_vit_h,
    "vit_h": build_pair_sam_vit_h,
    "vit_b": build_pair_sam_vit_b,
}

def _build_pair_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes=19,
    checkpoint=None,
    adapter_variant='reference',
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

    prompt_encoder = PairPromptEncoder(
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

    from .modeling import SimpleFPN, M2FDecoder, MSDeformAttnPixelDecoder
    simple_fpn = SimpleFPN(dim=prompt_embed_dim)
    pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=prompt_embed_dim, mask_dim=prompt_embed_dim)
    m2f_decoder = M2FDecoder(num_classes=num_classes, hidden_dim=prompt_embed_dim)

    # 2. 組合 PairSAM
    sam = PairSAM(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        fusion_module=fusion_module,
        text_encoder=text_encoder,
        num_classes=num_classes,
        simple_fpn=simple_fpn,
        m2f_decoder=m2f_decoder,
        pixel_decoder=pixel_decoder,
    )

    # 2b. [ablation B] 在載入 checkpoint 之前換好注入器，確保 sam_adapter 的已訓練
    # 權重（vgg_injector.adapters.*）能被下方 state_dict 過濾正確匹配並載入。
    if adapter_variant == 'sam_adapter':
        from .modeling.sam_adapter_injector import SameImageAdapterInjector
        _vit_dim = sam.image_encoder.patch_embed.proj.out_channels
        sam.vgg_injector = SameImageAdapterInjector(vit_dim=_vit_dim)
        sam.adapter_variant = 'sam_adapter'
        sam._adapter_reference_free = True

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