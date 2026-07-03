import torch
import pytest
from torch import nn
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


def test_zero_input_stays_finite():
    # 全零輸入的 degenerate 冒煙檢查：outputs_mask 恆為 0 → sigmoid=0.5 → 嚴格
    # < 0.5 全 False（attn_mask 全開），forward 不應產生 NaN/Inf。
    dec = M2FDecoder(num_classes=19)
    feats, mf = _fake_inputs()
    out = dec([f * 0 for f in feats], mf * 0, None, None)
    assert torch.isfinite(out["pred_masks"]).all()
    assert torch.isfinite(out["pred_logits"]).all()


class _Query0FullyMasked(nn.Module):
    """替換 mask_embed：query 0 的 mask embedding 恆為 -1，其餘為 +1。

    搭配 mask_features=ones，einsum 沿 channel 求和 → query 0 的 outputs_mask
    在所有空間位置皆為 -256（雙線性插值後仍是常數負值），sigmoid≈0 < 0.5 →
    attn_mask 對 query 0 整列皆 True（全被遮罩）。這正是上游 fallback
    `attn_mask[sum(-1)==shape[-1]] = False` 要接住的「某 query 對某層無任何可
    attend 位置」情境；若無此 fallback，該 query 的 cross-attention softmax
    會對全 -inf 取值而產生 NaN。
    """

    def __init__(self, mask_dim=256):
        super().__init__()
        self.mask_dim = mask_dim

    def forward(self, decoder_output):
        B, Q, _ = decoder_output.shape
        me = torch.ones(B, Q, self.mask_dim)
        me[:, 0, :] = -1.0
        return me


def test_empty_attn_mask_fallback_no_nan():
    # 確定性逼出上游「某 query 的 attn_mask 整列全 True → fallback 全開」分支，
    # 並斷言：(a) 該 fallback 條件在 forward 過程中確實成立過；(b) 輸出仍為有限值。
    torch.manual_seed(0)
    dec = M2FDecoder(num_classes=19, dec_layers=2)
    feats, _ = _fake_inputs()
    mask_features = torch.ones(1, 256, 256, 256)  # einsum 退化為 mask_embed 沿 C 求和
    dec.mask_embed = _Query0FullyMasked(mask_dim=256)

    # 攔截 forward_prediction_heads 回傳的 attn_mask，確認 fallback 條件確實觸發
    captured = []
    orig = dec.forward_prediction_heads

    def spy(output, mf_, attn_mask_target_size):
        outputs_class, outputs_mask, attn_mask = orig(
            output, mf_, attn_mask_target_size)
        captured.append(attn_mask)
        return outputs_class, outputs_mask, attn_mask

    dec.forward_prediction_heads = spy

    out = dec(feats, mask_features, None, None)

    # (a) 至少一次 head 呼叫中存在整列全 True 的 query（即 fallback 該接手的情境）
    fallback_hit = any(
        bool((am.sum(-1) == am.shape[-1]).any()) for am in captured
    )
    assert fallback_hit, "fallback 條件從未成立，此測試未涵蓋目標分支"
    # (b) 有 fallback 兜底，輸出保持有限（無 NaN/Inf）
    assert torch.isfinite(out["pred_masks"]).all()
    assert torch.isfinite(out["pred_logits"]).all()
