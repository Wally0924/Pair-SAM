"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_prior_source_switch.py -v

--prior_source 的語義鎖定：
  reference（預設）= UAWarpC 對齊後的跨視角參考先驗，既有行為
  self            = 當前影像的 VGG 多尺度先驗（ViT-Adapter 式 SPM），conf≡1

_build_adapter_prior 以 stub 直測，避免建構完整 ViT-H 前向（CPU 上過慢）。
builder 測試沿用 tests/test_build_from_config.py 的 vit_b + checkpoint=None 慣例（約 4s）。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

import torch

from segment_anything.modeling.pair_sam import PairSAM
from segment_anything.build_pair_sam import build_pair_sam_from_config


class _RecordingFusion:
    """記錄呼叫了哪個方法，回傳形狀無關緊要的假先驗。"""

    def __init__(self):
        self.calls = []

    def pre_align(self, img_curr, img_ref, out_size, l2_native=False):
        self.calls.append('pre_align')
        return {'l2': torch.zeros(1), 'l3': torch.zeros(1),
                'l4': torch.zeros(1), 'mask': torch.zeros(1)}

    def self_prior(self, img_curr, out_size, l2_native=False):
        self.calls.append('self_prior')
        return {'l2': torch.zeros(1), 'l3': torch.zeros(1), 'l4': torch.zeros(1)}


def _stub(prior_source='reference'):
    """以未實例化 PairSAM 的方式直測 _build_adapter_prior 的分派邏輯。"""
    obj = SimpleNamespace(
        use_vgg_adapter=True,
        _adapter_reference_free=False,
        prior_source=prior_source,
        fusion_module=_RecordingFusion(),
        device=torch.device('cpu'),
        image_encoder=SimpleNamespace(
            img_size=1024,
            patch_embed=SimpleNamespace(proj=SimpleNamespace(stride=(16, 16))),
        ),
    )
    obj._build_adapter_prior = PairSAM._build_adapter_prior.__get__(obj)
    return obj


def _batched(with_clear=True):
    rec = {'image': torch.zeros(3, 1024, 1024)}
    if with_clear:
        rec['clear_image'] = torch.zeros(3, 1024, 1024)
    return [rec]


def test_default_prior_source_dispatches_to_pre_align():
    obj = _stub('reference')
    out = obj._build_adapter_prior(_batched())
    assert obj.fusion_module.calls == ['pre_align']
    assert 'mask' in out


def test_self_prior_source_dispatches_to_self_prior():
    obj = _stub('self')
    out = obj._build_adapter_prior(_batched())
    assert obj.fusion_module.calls == ['self_prior']
    assert 'mask' not in out


def test_self_mode_does_not_require_clear_image():
    """self 模式不需要參考影像；缺 clear_image 仍須建出先驗。"""
    obj = _stub('self')
    out = obj._build_adapter_prior(_batched(with_clear=False))
    assert obj.fusion_module.calls == ['self_prior']
    assert out is not None


def test_reference_mode_without_clear_image_returns_none():
    """reference 模式缺 clear_image 時維持既有行為：不注入先驗。"""
    obj = _stub('reference')
    out = obj._build_adapter_prior(_batched(with_clear=False))
    assert out is None
    assert obj.fusion_module.calls == []


def test_precomputed_embedding_skips_prior():
    """已有 image_embedding 時不重建先驗（既有行為）。"""
    obj = _stub('self')
    out = obj._build_adapter_prior([{'image_embedding': torch.zeros(1),
                                     'image': torch.zeros(3, 1024, 1024)}])
    assert out is None
    assert obj.fusion_module.calls == []


def _cfg(**over):
    base = dict(model_type='vit_b', use_vgg_adapter=True, inject='pre',
                decoder='unified', lrh=True, mfb=True, ref=True)
    base.update(over)
    return base


def test_builder_default_is_reference():
    m = build_pair_sam_from_config(_cfg(), checkpoint=None)
    assert m.prior_source == 'reference'


def test_builder_maps_prior_source_self():
    """關鍵：eval 與 test dump 都經 load_pair_sam_from_ablation → build_pair_sam_from_config
    重建模型。若此映射缺失，評估會靜默地以 reference 模式跑，得到錯誤數字。"""
    m = build_pair_sam_from_config(_cfg(prior_source='self'), checkpoint=None)
    assert m.prior_source == 'self'


def test_builder_rejects_unknown_prior_source():
    try:
        build_pair_sam_from_config(_cfg(prior_source='bogus'), checkpoint=None)
    except ValueError as e:
        assert 'prior_source' in str(e)
    else:
        raise AssertionError("未知 prior_source 必須拋 ValueError")


def test_self_mode_does_not_set_adapter_reference_free():
    """_adapter_reference_free 是 sam_adapter(W4) 專用，會整個替換 vgg_injector。
    self 模式必須保留 RPM 路徑。"""
    m = build_pair_sam_from_config(_cfg(prior_source='self'), checkpoint=None)
    assert m._adapter_reference_free is False
    assert hasattr(m.vgg_injector, 'rpm')


def test_self_mode_keeps_zero_init_gate():
    """閘控必須與 FULL 相同的零初始化，否則 P1 與 FULL 之間多一個變因。"""
    m = build_pair_sam_from_config(_cfg(prior_source='self'), checkpoint=None)
    for inj in m.vgg_injector.injectors:
        assert torch.equal(inj.gamma.detach(), torch.zeros_like(inj.gamma))


def _train_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_train_cli_exposes_prior_source():
    """行為測試：--help 必須列出 --prior_source 及其兩個合法值。

    argparse parser 建構於 train.py 的 main() 內部、由 __main__ 守衛，
    無法 import 取得，故以子行程驗證真實的 argparse 行為。
    """
    import subprocess
    out = subprocess.run([sys.executable, 'train.py', '--help'],
                         cwd=_train_dir(), capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert '--prior_source' in out.stdout, "train.py 缺 --prior_source 旗標"
    assert '{reference,self}' in out.stdout, \
        "--prior_source 必須限定 choices=['reference', 'self']"


def test_train_cli_rejects_invalid_prior_source():
    """行為測試：非法值必須被 argparse 擋下，而非靜默落入預設。"""
    import subprocess
    out = subprocess.run([sys.executable, 'train.py', '--prior_source', 'bogus'],
                         cwd=_train_dir(), capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 2, f"預期 argparse 錯誤退出碼 2，實得 {out.returncode}"
    assert 'invalid choice' in out.stderr


def test_train_wires_prior_source_into_abl_cfg():
    """接線檢查：abl_cfg 必須含 prior_source=args.prior_source。

    這是原始碼層級斷言，不是行為測試 —— 刻意如此。要在執行期驗證需要
    完整的 CUDA 環境、資料集與 checkpoint，成本過高；真正的執行期驗證是
    Task 4 Step 2（訓練啟動兩分鐘後檢查 ablation_config.json）。

    保留本項的理由：若 abl_cfg 遺漏此鍵，eval 會從 ablation_config.json
    重建模型時靜默退回 reference 模式、產出錯誤數字。此斷言讓接線遺漏在
    改動 train.py 的當下就被發現，而不是八小時訓練跑完後才發現。
    """
    import re
    with open(os.path.join(_train_dir(), 'train.py'), encoding='utf-8') as f:
        src = f.read()
    assert re.search(r'prior_source\s*=\s*args\.prior_source', src), \
        "abl_cfg 必須含 prior_source=args.prior_source"
