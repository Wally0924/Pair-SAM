"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_self_prior.py -v

self_prior() 的語義鎖定：先驗取自當前影像、不經 UAWarpC、無 'mask' 鍵
（使 ReferencePriorModule 走既有 no-mask 分支取得 conf≡1）。

以 256×256 輸入 + out_size=(16,16) 讓 VGG16 maxpool 管線成立且 CPU 執行時間短，
與 tests/test_pre_align_native_l2.py 同一慣例。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from segment_anything.modeling.fusion import CMAAlignment
from segment_anything.modeling.deform_adapter import ReferencePriorModule


def make_model():
    """隨機初始化的 CMAAlignment —— 只驗形狀與語義，不需預訓練權重。"""
    return CMAAlignment(embed_dim=256, pretrained_path=None)


def make_image(H=256, W=256, B=1):
    return torch.rand(B, 3, H, W) * 255.0


def test_self_prior_shapes_match_pre_align():
    """三個尺度的 shape 與 channel 必須逐一等同 pre_align(l2_native=True)。"""
    model = make_model()
    img_curr, img_ref = make_image(), make_image()
    ref_out = model.pre_align(img_curr, img_ref, out_size=(16, 16), l2_native=True)
    self_out = model.self_prior(img_curr, out_size=(16, 16), l2_native=True)
    for k in ('l2', 'l3', 'l4'):
        assert self_out[k].shape == ref_out[k].shape, (
            f"{k} shape 不對等：self={self_out[k].shape} vs ref={ref_out[k].shape}")


def test_self_prior_omits_mask_key():
    """不得回傳 'mask'；RPM 靠 feats.get('mask', None) is None 取得 conf≡1。"""
    model = make_model()
    self_out = model.self_prior(make_image(), out_size=(16, 16), l2_native=True)
    assert 'mask' not in self_out


def test_self_prior_uses_current_image_features():
    """l3 必須等於當前影像 VGG index-3 特徵縮放後的結果，不含任何翹曲。"""
    model = make_model()
    img = make_image()
    with torch.no_grad():
        feats, _ = model._extract_vgg_features(img)
        expected = F.interpolate(feats[3], size=(16, 16),
                                 mode='bilinear', align_corners=False)
    out = model.self_prior(img, out_size=(16, 16), l2_native=True)
    assert torch.allclose(out['l3'], expected, atol=1e-5)


def test_self_prior_is_independent_of_reference_image():
    """同一張當前影像 → 輸出恆定，與是否存在參考影像無關。"""
    model = make_model()
    img = make_image()
    a = model.self_prior(img, out_size=(16, 16), l2_native=True)
    b = model.self_prior(img, out_size=(16, 16), l2_native=True)
    assert torch.allclose(a['l2'], b['l2'], atol=1e-6)


def test_self_prior_sets_neutral_telemetry():
    """pair_trainer.py 讀 _last_conf_mean/_last_valid_ratio；self 模式須為中性值 1.0，
    否則 train_log.csv 欄位語義與其他 run 不一致。"""
    model = make_model()
    model._last_conf_mean = 0.3
    model._last_valid_ratio = 0.3
    model._last_flow = torch.zeros(1, 2, 16, 16)
    model._last_confidence_map = torch.zeros(1, 1, 16, 16)
    model.self_prior(make_image(), out_size=(16, 16), l2_native=True)
    assert model._last_conf_mean == 1.0
    assert model._last_valid_ratio == 1.0
    assert model._last_flow is None
    assert model._last_confidence_map is None


def test_rpm_returns_neutral_conf_without_mask_key():
    """契約鎖定：feats 無 'mask' 鍵 → conf≡1 且參考特徵 c 保持非零。
    這是 deform_adapter.py 不需修改的原因。"""
    torch.manual_seed(0)
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=8, l4_channels=8, dim=16)
    feats = {
        'l2': torch.randn(1, 8, 8, 8),
        'l3': torch.randn(1, 8, 4, 4),
        'l4': torch.randn(1, 8, 2, 2),
    }
    c, conf = rpm(feats)
    assert torch.equal(conf, torch.ones_like(conf)), "無 mask 鍵時 conf 必須≡1"
    assert c.abs().sum() > 0, "先驗特徵不得被歸零（那是 --no-ref 的語義）"


def _small_rpm_and_feats():
    torch.manual_seed(0)
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=8, l4_channels=8, dim=16)
    feats = {
        'l2': torch.randn(1, 8, 8, 8),
        'l3': torch.randn(1, 8, 4, 4),
        'l4': torch.randn(1, 8, 2, 2),
    }
    return rpm, feats


def test_rpm_projections_receive_gradient_without_mask():
    """梯度連通：self 模式下 proj_c2/c3/c4 必須收到非零梯度。

    這是 self 與 --no-ref 的關鍵區別。--no-ref 走 c = zeros_like(c)，
    proj 卷積的輸出被丟棄、梯度斷聯（Phase 5 腳本記載的良性 Grad Audit 警報）；
    self 模式的先驗特徵實際參與注入，斷聯即代表接線錯誤。
    """
    rpm, feats = _small_rpm_and_feats()
    c, conf = rpm(feats)
    (c * conf).sum().backward()
    for name in ('proj_c2', 'proj_c3', 'proj_c4'):
        grad = getattr(rpm, name).weight.grad
        assert grad is not None, f"{name} 未收到梯度"
        assert grad.abs().sum() > 0, f"{name} 梯度全為零"


def test_no_ref_projections_stay_disconnected():
    """對照鎖定：use_reference=False 時 proj 卷積梯度斷聯，確認上一個測試
    量到的是真實差異而非恆真斷言。"""
    rpm, feats = _small_rpm_and_feats()

    # 先驗證 use_reference=True 時梯度確實流通
    rpm.use_reference = True
    c_true, conf = rpm(feats)
    (c_true * conf).sum().backward()
    assert rpm.proj_c2.weight.grad is not None and rpm.proj_c2.weight.grad.abs().sum() > 0
    rpm.zero_grad()

    # 再驗證 use_reference=False 時梯度斷聯（c 全零，計算圖被切斷）
    rpm.use_reference = False
    c_false, conf = rpm(feats)
    # use_reference=False 時，c 應為全零
    assert c_false.abs().sum() == 0
    # c 沒有梯度計算圖，因此 proj 層不會收到梯度
    assert not c_false.requires_grad or c_false.grad_fn is None
    grad = rpm.proj_c2.weight.grad
    assert grad is None or grad.abs().sum() == 0
