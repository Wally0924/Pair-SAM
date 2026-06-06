# segment-anything/tests/test_lovasz_schedule.py
"""
延後啟動 Lovász-Softmax 的排程邏輯（--lovasz_start_epoch）。
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_lovasz_schedule.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.new_loss import lovasz_weight_for_epoch


def test_disabled_before_start_epoch():
    # start=5：前 5 個 epoch（index 0..4）Lovász 停用 → 權重 0
    assert lovasz_weight_for_epoch(epoch_index=0, start_epoch=5, target_weight=1.0) == 0.0
    assert lovasz_weight_for_epoch(epoch_index=4, start_epoch=5, target_weight=1.0) == 0.0


def test_active_from_start_epoch_onward():
    # epoch_index >= start → 回傳目標權重
    assert lovasz_weight_for_epoch(epoch_index=5, start_epoch=5, target_weight=1.0) == 1.0
    assert lovasz_weight_for_epoch(epoch_index=40, start_epoch=5, target_weight=0.75) == 0.75


def test_start_zero_is_backward_compatible():
    # start=0（預設）→ 從第一個 epoch 起就啟用，等同舊行為
    assert lovasz_weight_for_epoch(epoch_index=0, start_epoch=0, target_weight=1.0) == 1.0
