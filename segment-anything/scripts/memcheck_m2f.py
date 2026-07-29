# segment-anything/scripts/memcheck_m2f.py
"""ViT-H + DeformAdapter + SimpleFPN + M2FDecoder + M2FSetLoss 全路徑峰值顯存量測。
通過標準：train step（forward + loss + backward）峰值 ≤ 20 GB（4090 24GB 留 buffer）。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from segment_anything.build_pair_sam import build_pair_sam_from_config
from utils.m2f_loss import M2FSetLoss

cfg = {"model_type": "vit_h", "use_vgg_adapter": True, "decoder": "m2f"}
model = build_pair_sam_from_config(cfg).cuda().train()
# 對齊真實訓練（train.py:301 / pretrain_cityscapes.py:109）：ViT-H block gradient
# checkpointing。少了這行等於測「不切實際的無 checkpointing 設定」，必 OOM。
model.image_encoder.use_checkpoint = True
crit = M2FSetLoss().cuda()

names = [n for n, _ in sorted(model.CLASS_MAP.items(), key=lambda kv: kv[1])]
batched = [{
    "image": torch.randint(0, 255, (3, 1024, 1024), dtype=torch.float32, device="cuda"),
    "clear_image": torch.randint(0, 255, (3, 1024, 1024), dtype=torch.float32, device="cuda"),
    "original_size": (1024, 1024),
    "text_prompts": names,
    "condition_id": torch.tensor(2),
}]
gt = torch.randint(0, 19, (1, 1024, 1024), device="cuda")

scaler = torch.amp.GradScaler("cuda", init_scale=2048)
torch.cuda.reset_peak_memory_stats()
for step in range(3):
    with torch.amp.autocast("cuda"):
        out = model(batched)[0]
        loss, log = crit(out, gt)
    scaler.scale(loss).backward()
    model.zero_grad(set_to_none=True)
    print(f"step {step}: loss={float(loss):.3f} cls={log['cls']:.3f} "
          f"bce={log['bce']:.3f} dice={log['dice']:.3f}")

peak = torch.cuda.max_memory_allocated() / 2**30
print(f"peak allocated: {peak:.2f} GiB")
assert peak <= 20.0, f"顯存超標: {peak:.2f} GiB > 20 GiB"
assert torch.isfinite(loss), "loss 出現 NaN/Inf"
print("MEMCHECK PASS")
