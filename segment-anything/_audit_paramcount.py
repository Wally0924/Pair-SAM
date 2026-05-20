"""Audit parameter counts for Table 1 verification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segment_anything.build_weather_sam import build_weather_sam_vit_h
import torch

model = build_weather_sam_vit_h(checkpoint=None)

def count(mod, only_trainable=False):
    if mod is None: return 0
    if only_trainable:
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return sum(p.numel() for p in mod.parameters())

# Mimic trainer freezing logic from weather_trainer.py:127-198
for p in model.parameters():
    p.requires_grad = False

main_lr_modules = [
    model.condition_encoder,
    model.text_encoder.projection,
    model.context_fusion_head,
]
decoder_lr_modules = [
    model.mask_decoder.output_upscaling,
    model.mask_decoder.class_mask_tokens,
    model.mask_decoder.class_hypernetworks_mlps,
]
decoder_transformer_modules = [
    model.mask_decoder.transformer,
]

for mod in main_lr_modules + decoder_lr_modules + decoder_transformer_modules:
    for p in mod.parameters():
        p.requires_grad = True

for p in model.fusion_module.parameters():
    p.requires_grad = False

model.pe_layer.requires_grad = True

model.enable_vgg_adapter('pre')
for p in model.vgg_injector.parameters():
    p.requires_grad = True

print('=== MODULE BREAKDOWN ===')
ie_t, ie = count(model.image_encoder, True), count(model.image_encoder)
print(f'(1) SAM ViT-H image encoder:    total={ie/1e6:7.2f}M  trainable={ie_t/1e6:7.4f}M')

te_t, te = count(model.text_encoder, True), count(model.text_encoder)
te_proj = count(model.text_encoder.projection)
print(f'(2) CLIP text encoder (+ proj): total={te/1e6:7.2f}M  trainable={te_t/1e6:7.4f}M  (proj only={te_proj/1e6:7.4f}M)')

fm_t, fm = count(model.fusion_module, True), count(model.fusion_module)
print(f'(3) CMAAlignment (VGG+UAWarpC): total={fm/1e6:7.2f}M  trainable={fm_t/1e6:7.4f}M')

vgi_t, vgi = count(model.vgg_injector, True), count(model.vgg_injector)
print(f'(4) Cross-attention adapter:    total={vgi/1e6:7.2f}M  trainable={vgi_t/1e6:7.4f}M')

mdt_t, mdt = count(model.mask_decoder.transformer, True), count(model.mask_decoder.transformer)
print(f'(5) TwoWayTransformer:          total={mdt/1e6:7.2f}M  trainable={mdt_t/1e6:7.4f}M')

# Class tokens + hypernets + upscaling
ct_t = count(model.mask_decoder.class_mask_tokens, True)
ct = count(model.mask_decoder.class_mask_tokens)
hn_t = count(model.mask_decoder.class_hypernetworks_mlps, True)
hn = count(model.mask_decoder.class_hypernetworks_mlps)
up_t = count(model.mask_decoder.output_upscaling, True)
up = count(model.mask_decoder.output_upscaling)
mdh_t = ct_t + hn_t + up_t
mdh = ct + hn + up
print(f'(6) Class tokens+hypernets+up:  total={mdh/1e6:7.2f}M  trainable={mdh_t/1e6:7.4f}M')
print(f'    (class_tokens={ct/1e3:.1f}K, hypernets={hn/1e6:.3f}M, upscaling={up/1e6:.3f}M)')

pel = model.pe_layer.numel()
print(f'(7) pe_layer:                   total={pel/1e6:7.4f}M  trainable={pel/1e6 if model.pe_layer.requires_grad else 0:7.4f}M')

cfh_t = count(model.context_fusion_head, True)
cfh = count(model.context_fusion_head)
ce_t = count(model.condition_encoder, True)
ce = count(model.condition_encoder)
pe_enc = count(model.prompt_encoder)
pe_enc_t = count(model.prompt_encoder, True)
small = cfh_t + ce_t + pe_enc_t
print(f'(8) Refinement+cond+prompt enc: total={(cfh+ce+pe_enc)/1e3:7.2f}K trainable={small/1e3:7.4f}K')
print(f'    (refinement={cfh/1e3:.2f}K, condition={ce/1e3:.2f}K, prompt_enc={pe_enc/1e3:.2f}K)')

print('=== TOTAL ===')
total = count(model)
total_t = count(model, True)
print(f'Total model:     {total/1e6:.2f}M ({total:,} params)')
print(f'Total trainable: {total_t/1e6:.2f}M ({total_t:,} params)')
print(f'Total frozen:    {(total-total_t)/1e6:.2f}M')
print(f'Trainable %:     {total_t/total*100:.3f}%')
