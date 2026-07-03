import torch
import pytest
from segment_anything.modeling.m2f_decoder import M2FDecoder


def _fake_inputs(B=1):
    feats = [
        torch.randn(B, 256, 32, 32),
        torch.randn(B, 256, 64, 64),
        torch.randn(B, 256, 128, 128),
    ]
    mask_features = torch.randn(B, 256, 256, 256)
    return feats, mask_features


def test_output_contract():
    dec = M2FDecoder(num_classes=19, hidden_dim=256, dec_layers=9)
    feats, mf = _fake_inputs()
    text = torch.randn(19, 256)
    cond = torch.randn(1, 1, 256)
    out = dec(feats, mf, text, cond)
    assert tuple(out["pred_logits"].shape) == (1, 19, 20)   # 19 類 + no-object
    assert tuple(out["pred_masks"].shape) == (1, 19, 256, 256)
    assert len(out["aux_outputs"]) == 9                      # initial + 前 8 層
    for aux in out["aux_outputs"]:
        assert tuple(aux["pred_logits"].shape) == (1, 19, 20)
        assert tuple(aux["pred_masks"].shape) == (1, 19, 256, 256)


def test_optional_text_and_condition():
    dec = M2FDecoder(num_classes=19)
    feats, mf = _fake_inputs()
    out = dec(feats, mf, None, None)  # 兩者皆可省略（ablation 路徑）
    assert tuple(out["pred_masks"].shape) == (1, 19, 256, 256)


def test_condition_token_changes_predictions():
    torch.manual_seed(0)
    dec = M2FDecoder(num_classes=19).eval()
    feats, mf = _fake_inputs()
    with torch.no_grad():
        out_a = dec(feats, mf, None, torch.full((1, 1, 256), 1.0))
        out_b = dec(feats, mf, None, torch.full((1, 1, 256), -1.0))
    # condition token 經 self-attention 影響 class queries → 預測必須不同
    assert not torch.allclose(out_a["pred_masks"], out_b["pred_masks"])


def test_gradients_reach_all_inputs():
    dec = M2FDecoder(num_classes=19)
    feats, mf = _fake_inputs()
    for f in feats:
        f.requires_grad_(True)
    mf.requires_grad_(True)
    text = torch.randn(19, 256, requires_grad=True)
    out = dec(feats, mf, text, None)
    (out["pred_masks"].sum() + out["pred_logits"].sum()).backward()
    assert mf.grad is not None
    assert text.grad is not None
    for f in feats:
        assert f.grad is not None


def test_empty_attn_mask_fallback_no_nan():
    # 全零 mask_features → 每層 mask 預測全 0 → sigmoid=0.5 邊界，逼出上游
    # 「attn_mask 全空 → fallback 全開」的程式路徑，檢查數值穩定
    dec = M2FDecoder(num_classes=19)
    feats, mf = _fake_inputs()
    out = dec([f * 0 for f in feats], mf * 0, None, None)
    assert torch.isfinite(out["pred_masks"]).all()
    assert torch.isfinite(out["pred_logits"]).all()
