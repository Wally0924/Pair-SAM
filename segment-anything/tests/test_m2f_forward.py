import torch
import pytest
from segment_anything.build_pair_sam import build_pair_sam_from_config


@pytest.fixture(scope="module")
def model():
    cfg = {"model_type": "vit_b", "use_vgg_adapter": False, "decoder": "m2f"}
    m = build_pair_sam_from_config(cfg)
    return m.to("cuda" if torch.cuda.is_available() else "cpu")


def _fake_input(model):
    dev = model.device
    names = [n for n, _ in sorted(model.CLASS_MAP.items(), key=lambda kv: kv[1])]
    return [{
        "image": torch.randint(0, 255, (3, 1024, 1024), dtype=torch.float32, device=dev),
        "original_size": (1024, 1024),
        "text_prompts": names,
        "condition_id": torch.tensor(2),  # snow
    }]


def test_m2f_flag_set(model):
    assert model.decoder_arch == "m2f"
    assert model.use_lrh is False


def test_train_mode_output_contract(model):
    model.train()
    with torch.no_grad():  # 只驗輸出契約，省顯存
        out = model(_fake_input(model))[0]
    assert tuple(out["pred_logits"].shape) == (1, 19, 20)
    assert tuple(out["pred_masks"].shape) == (1, 19, 256, 256)
    assert len(out["aux_outputs"]) == 9
    assert out["class_ids"] == list(range(19))
    assert tuple(out["low_res_logits"].shape) == (1, 19, 256, 256)
    assert "masks" not in out  # 訓練模式不做 HR postprocess


def test_eval_mode_semantic_output(model):
    model.eval()
    with torch.no_grad():
        out = model(_fake_input(model))[0]
    assert tuple(out["masks"].shape) == (1, 19, 1024, 1024)
    # semantic map is a sum over queries of softmax·sigmoid → non-negative and finite,
    # but NOT bounded by 1 (bound is [0, num_queries]); only argmax'd downstream.
    assert out["low_res_logits"].min() >= 0
    assert torch.isfinite(out["low_res_logits"]).all()


def test_legacy_path_still_works():
    cfg = {"model_type": "vit_b", "use_vgg_adapter": False, "decoder": "unified"}
    m = build_pair_sam_from_config(cfg)
    assert m.decoder_arch == "legacy"
    assert m.mask_decoder.decoder_mode == "unified"
