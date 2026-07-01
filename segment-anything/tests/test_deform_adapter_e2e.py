"""端到端 smoke test：整個 WeatherSAM forward（含 DeformAdapter A3）輸出 key 與舊版一致。

GPU 測試：用真實 ViT-H（1024² 太慢，CPU skip）；在 CUDA 下以 bfloat16 autocast 執行。
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter_e2e.py -v
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.build_weather_sam import build_weather_sam_from_config


def test_full_model_forward_keys_unchanged():
    """整個模型 forward 輸出 key 集合不變，且 low_res_logits 全部有限。

    - cfg: use_vgg_adapter=True + ref=True → pre_align + inject/extract hook 全部觸發
    - clear_image 存在 → pre_align 路徑確實執行（_vgg_ref_aligned 不為 None）
    - 僅 GPU 執行（ViT-H at 1024² CPU 需數分鐘，效果同 hang）
    """
    if not torch.cuda.is_available():
        pytest.skip("ViT-H e2e requires GPU")
    # ViT-H global attention needs ~5 GB free (bf16 weights + 4 × 512MB attention matrices);
    # skip gracefully if memory is crowded (e.g. another training job on the same GPU).
    _free = torch.cuda.mem_get_info()[0]  # bytes free
    if _free < 5 * 1024 ** 3:
        pytest.skip(f"ViT-H e2e requires ≥5 GB free GPU memory; only {_free/1e9:.1f} GB free")

    cfg = {
        "model_type": "vit_h",
        "use_vgg_adapter": True,
        "cond": True,
        "lrh": True,
        "decoder": "unified",
        "ref": True,
        "mfb": True,
    }
    model = build_weather_sam_from_config(cfg, checkpoint=None).eval()
    model = model.cuda().to(torch.bfloat16)  # half precision to reduce peak VRAM

    S = model.image_encoder.img_size  # 1024

    batch = [
        {
            "image": torch.randint(0, 255, (3, S, S)).float().cuda(),
            "clear_image": torch.randint(0, 255, (3, S, S)).float().cuda(),
            "text_prompts": ["road", "car"],
            "condition_id": torch.tensor(0).cuda(),
            "original_size": (S, S),
        }
    ]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)

    assert set(out[0].keys()) == {"masks", "low_res_logits", "class_ids"}
    assert torch.isfinite(out[0]["low_res_logits"].float()).all()
