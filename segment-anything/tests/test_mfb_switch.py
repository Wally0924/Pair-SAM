# segment-anything/tests/test_mfb_switch.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_mfb_switch.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.new_loss import ContextLoss


def test_mfb_on_uses_nonuniform_weights():
    loss = ContextLoss(use_mfb=True)
    w = loss.ce_loss_fn.weight
    assert w is not None
    assert not torch.allclose(w, torch.ones_like(w))  # MFB 權重非均勻


def test_mfb_off_uses_uniform_weights():
    loss = ContextLoss(use_mfb=False)
    w = loss.ce_loss_fn.weight
    assert w is None or torch.allclose(w, torch.ones_like(w))  # uniform


def test_mfb_default_on_backward_compatible():
    loss = ContextLoss()  # 預設 = FULL 行為
    w = loss.ce_loss_fn.weight
    assert w is not None and not torch.allclose(w, torch.ones_like(w))
