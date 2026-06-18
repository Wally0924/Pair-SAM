"""
P1（--cond）開關測試，對應 spec §2.7 / §5.1。

執行：conda run -n sam_env python -m pytest segment-anything/tests/test_cond_switch.py -v

設計同 A2 的 --ref：cond off 時將 condition_id 固定為共享索引 0，
ConditionEncoder 退化為「與天氣條件無關的可學習常數」，保參數路徑不變
（隔離「條件資訊 vs 容量」）。用 vit_b + checkpoint=None 輕量建模（約 4s）。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import functools
import torch
from segment_anything.build_weather_sam import build_weather_sam_from_config


def _cfg(**over):
    base = dict(model_type='vit_b', use_vgg_adapter=True, inject='pre',
                decoder='unified', lrh=True, mfb=True, ref=True, cond=True)
    base.update(over)
    return base


@functools.lru_cache(maxsize=None)
def _model(cond=True):
    return build_weather_sam_from_config(_cfg(cond=cond), checkpoint=None)


def _cid(i):
    return torch.tensor(i, dtype=torch.long)


def test_use_cond_default_true():
    assert _model(cond=True).use_cond is True


def test_config_maps_cond_off():
    assert _model(cond=False).use_cond is False


def test_cond_off_insensitive_to_condition_id():
    m = _model(cond=True)
    m.use_cond = False
    with torch.no_grad():
        a = m._encode_condition(_cid(0))
        b = m._encode_condition(_cid(3))
    assert torch.allclose(a, b, atol=1e-6)   # cond off：固定索引 0 → 不同條件輸出相同


def test_cond_on_sensitive_to_condition_id():
    m = _model(cond=True)
    m.use_cond = True
    with torch.no_grad():
        a = m._encode_condition(_cid(0))
        b = m._encode_condition(_cid(3))
    assert not torch.allclose(a, b, atol=1e-6)  # cond on：不同條件 → 不同輸出


def test_param_count_same_on_vs_off():
    n_on = sum(p.numel() for p in _model(cond=True).parameters())
    n_off = sum(p.numel() for p in _model(cond=False).parameters())
    assert n_on == n_off
