import torch
import pytest
from utils.m2f_loss import M2FSetLoss, point_sample


def _fake_output(num_aux=2, device="cpu"):
    def one():
        return {
            "pred_logits": torch.randn(1, 19, 20, device=device),
            "pred_masks": torch.randn(1, 19, 256, 256, device=device),
        }
    out = one()
    out["aux_outputs"] = [one() for _ in range(num_aux)]
    return out


def _fake_gt(present=(0, 13), device="cpu"):
    gt = torch.full((1, 1024, 1024), 255, dtype=torch.long, device=device)
    gt[:, :512, :] = present[0]
    gt[:, 512:, :] = present[1]
    return gt


def test_loss_is_finite_scalar_and_logs():
    crit = M2FSetLoss(num_classes=19, num_points=256)  # 測試用少量點
    loss, log = crit(_fake_output(), _fake_gt())
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert set(log) == {"cls", "bce", "dice"}


def test_gradient_flows():
    crit = M2FSetLoss(num_classes=19, num_points=256)
    out = _fake_output()
    out["pred_masks"].requires_grad_(True)
    out["pred_logits"].requires_grad_(True)
    loss, _ = crit(out, _fake_gt())
    loss.backward()
    assert out["pred_masks"].grad is not None
    assert out["pred_logits"].grad is not None


def test_absent_class_targets_no_object():
    # 只有 class 0 與 13 存在 → 其餘 17 個 query 的分類 target 是 no-object(19)。
    crit = M2FSetLoss(num_classes=19, num_points=256)
    out = _fake_output(num_aux=0)
    with torch.no_grad():
        out["pred_logits"].zero_()
        out["pred_logits"][..., 19] = 10.0  # 全部預測 no-object
    loss_all_noobj, _ = crit(out, _fake_gt())
    with torch.no_grad():
        out["pred_logits"][0, 0].zero_(); out["pred_logits"][0, 0, 0] = 10.0
        out["pred_logits"][0, 13].zero_(); out["pred_logits"][0, 13, 13] = 10.0
    loss_correct, _ = crit(out, _fake_gt())
    assert loss_correct < loss_all_noobj  # 答對 present 類必須讓 loss 下降


def test_all_ignore_image_no_mask_loss_no_nan():
    crit = M2FSetLoss(num_classes=19, num_points=256)
    gt = torch.full((1, 1024, 1024), 255, dtype=torch.long)
    loss, log = crit(_fake_output(num_aux=0), gt)
    assert torch.isfinite(loss)
    assert log["bce"] == 0.0 and log["dice"] == 0.0


def test_deep_supervision_scales_with_aux():
    torch.manual_seed(0)
    crit = M2FSetLoss(num_classes=19, num_points=256)
    out0 = _fake_output(num_aux=0)
    torch.manual_seed(0)
    out4 = _fake_output(num_aux=0)
    out4["aux_outputs"] = [
        {"pred_logits": out4["pred_logits"].clone(), "pred_masks": out4["pred_masks"].clone()}
        for _ in range(4)
    ]
    l0, _ = crit(out0, _fake_gt())
    l4, _ = crit(out4, _fake_gt())
    assert torch.allclose(l4, l0 * 5, rtol=0.05)  # 相同預測 ×5 組 → loss ≈ 5 倍


def test_point_sample_matches_grid_values():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    # 採樣像素中心 (0.125, 0.125) → 應取到左上角 0
    pts = torch.tensor([[[0.125, 0.125]]])
    v = point_sample(x, pts)
    assert torch.allclose(v[0, 0, 0], torch.tensor(0.0), atol=1e-4)
