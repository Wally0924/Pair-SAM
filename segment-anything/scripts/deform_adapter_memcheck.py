"""4090 記憶體 dry-run：1024²、bf16、grad ckpt、3 步 backward，印峰值記憶體。
執行：conda run -n sam_env python segment-anything/scripts/deform_adapter_memcheck.py
若 OOM，依 spec §7 緩解階梯：with_cp → deform_ratio → 1/8 預 pool → 退 2 尺度。

Mitigation log（依階梯順序填寫，無需改動則留 None）:
  (a) with_cp:       enabled (model.image_encoder.use_checkpoint = True, training mode)
  (b) deform_ratio:  0.5 (already default in DeformAdapter)
  (c) 1/8 pre-pool:  None — not needed
  (d) 2 scales:      None — not needed
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from segment_anything.build_weather_sam import build_weather_sam_from_config

cfg = {
    "model_type": "vit_h",
    "use_vgg_adapter": True,
    "cond": True,
    "lrh": True,
    "decoder": "unified",
    "ref": True,
    "mfb": True,
}

# Resolve checkpoint path relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT = os.path.join(_SCRIPT_DIR, "..", "checkpoints", "sam_vit_h_4b8939.pth")

print(f"Loading model from {_CHECKPOINT}...")
model = build_weather_sam_from_config(cfg, checkpoint=_CHECKPOINT).cuda()

# §7 mitigation (a): gradient checkpointing on ViT-H blocks
# Only active during training (use_ckpt = getattr(self, 'use_checkpoint', False) and self.training)
model.image_encoder.use_checkpoint = True
model.train()  # enable gradient checkpointing (use_checkpoint is gated on self.training)

S = model.image_encoder.img_size  # 1024

batch = [
    {
        "image": torch.randint(0, 255, (3, S, S)).float().cuda(),
        "clear_image": torch.randint(0, 255, (3, S, S)).float().cuda(),
        "text_prompts": ["road", "car", "person", "building"],
        "condition_id": torch.tensor(0).cuda(),
        "original_size": (S, S),
    }
]

torch.cuda.reset_peak_memory_stats()

for step in range(3):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
        loss = out[0]["low_res_logits"].float().mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[step {step}] peak {peak:.2f} GB")

print("OK: no OOM")
