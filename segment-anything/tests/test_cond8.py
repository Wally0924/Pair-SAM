"""8 條件（MUSES weather×time_of_day 全交叉）支援測試。

執行：conda run -n sam_env python -m pytest segment-anything/tests/test_cond8.py -v

涵蓋三件事：
1. num_conditions 正確傳到 condition_encoder，且預設 4 維持向後相容。
2. 4→8 擴表：ACDC 已學到的 0-3 列必須逐位元保留，4-7 列依語意組合初始化。
   （這是最容易靜默出錯的地方——builder 的 state_dict 過濾會直接丟棄 shape
   不符的鍵，若無擴表邏輯，整張表會退化為隨機初始化。）
3. dataloader 的 condition_id 範圍檢查隨 num_conditions 連動。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import functools
import pandas as pd
import pytest
import torch

from segment_anything.build_pair_sam import (
    build_pair_sam_from_config,
    _expand_condition_embedding,
    _COND_EXPAND_4_TO_8,
    _COND_KEY,
)
from utils.pair_dataloader import PairSegmentationDataset


def _cfg(**over):
    base = dict(model_type='vit_b', use_vgg_adapter=True, inject='pre',
                decoder='unified', lrh=True, mfb=True, ref=True, cond=True)
    base.update(over)
    return base


@functools.lru_cache(maxsize=None)
def _model(num_conditions=4):
    return build_pair_sam_from_config(
        _cfg(num_conditions=num_conditions), checkpoint=None)


# ── 1. num_conditions 傳遞與向後相容 ──────────────────────────────

def test_default_num_conditions_is_4():
    """未指定時維持 ACDC 的 4 條件，確保既有訓練腳本行為不變。"""
    m = _model()
    assert m.num_conditions == 4
    assert m.condition_encoder.num_embeddings == 4


def test_num_conditions_8_builds_8_row_table():
    m = _model(num_conditions=8)
    assert m.num_conditions == 8
    assert m.condition_encoder.num_embeddings == 8


def test_all_8_ids_encodable():
    """8 個 id 都要能查表且輸出彼此不同（每格都是獨立可學習條件）。"""
    m = _model(num_conditions=8)
    m.use_cond = True
    with torch.no_grad():
        feats = [m._encode_condition(torch.tensor(i, dtype=torch.long))
                 for i in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.allclose(feats[i], feats[j], atol=1e-6), \
                f'id {i} 與 {j} 的 condition feature 相同'


# ── 2. 4→8 擴表搬移 ───────────────────────────────────────────────

def _fake_ckpt(n_rows=4, dim=256):
    """造一張可辨識的假 embedding：第 i 列全為 (i+1)，便於驗證搬移結果。"""
    w = torch.arange(1, n_rows + 1, dtype=torch.float32).unsqueeze(1).repeat(1, dim)
    return {_COND_KEY: w, 'some.other.weight': torch.zeros(3)}


def test_expand_preserves_acdc_rows_bitwise():
    """0-3 列必須逐位元等於原 checkpoint —— ACDC 學到的天氣表徵不可被擾動。"""
    sd = _fake_ckpt()
    out = _expand_condition_embedding(sd, num_conditions=8)
    assert out[_COND_KEY].shape == (8, 256)
    assert torch.equal(out[_COND_KEY][:4], sd[_COND_KEY])


def test_expand_new_rows_are_source_means():
    sd = _fake_ckpt()
    old = sd[_COND_KEY]
    out = _expand_condition_embedding(sd, num_conditions=8)[_COND_KEY]
    for tgt, srcs in _COND_EXPAND_4_TO_8.items():
        assert torch.allclose(out[tgt], old[srcs].mean(dim=0)), \
            f'第 {tgt} 列未依 {srcs} 平均初始化'


def test_expand_does_not_mutate_input():
    """回傳淺拷貝，呼叫端傳入的 state_dict 不可被就地修改。"""
    sd = _fake_ckpt()
    _expand_condition_embedding(sd, num_conditions=8)
    assert sd[_COND_KEY].shape == (4, 256)


def test_expand_noop_when_sizes_match():
    sd = _fake_ckpt()
    assert _expand_condition_embedding(sd, num_conditions=4) is sd


def test_expand_noop_when_key_absent():
    sd = {'some.other.weight': torch.zeros(3)}
    assert _expand_condition_embedding(sd, num_conditions=8) is sd


def test_expanded_rows_survive_builder_shape_filter():
    """端到端：擴表後的權重必須真的進到模型，而非被 shape 過濾丟棄。"""
    m = _model(num_conditions=8)
    sd = _fake_ckpt()
    expanded = _expand_condition_embedding(sd, num_conditions=8)
    model_dict = m.state_dict()
    kept = {k: v for k, v in expanded.items()
            if k in model_dict and v.shape == model_dict[k].shape}
    assert _COND_KEY in kept, 'condition_encoder 被 builder 的 shape 過濾丟棄'


# ── 3. dataloader 範圍檢查連動 ────────────────────────────────────

MUSES_COND8_VAL = '/home/rvl1421/SAM_research-1/Datasets/muses_cond8_ref_rgb_val.csv'


@pytest.mark.skipif(not os.path.exists(MUSES_COND8_VAL),
                    reason='需先產生 muses_cond8_ref_rgb_val.csv')
def test_cond8_csv_rejected_under_num_conditions_4():
    """8 類 CSV 誤用 4 條件設定時必須明確報錯，不可靜默截斷。"""
    ds = PairSegmentationDataset(MUSES_COND8_VAL, mode='val', num_conditions=4)
    bad = ds.data.index[ds.data['condition_id'] >= 4].tolist()
    assert bad, '測試資料應含 id >= 4 的樣本'
    with pytest.raises(ValueError, match='越界'):
        ds[ds.data.index.get_loc(bad[0])]


@pytest.mark.skipif(not os.path.exists(MUSES_COND8_VAL),
                    reason='需先產生 muses_cond8_ref_rgb_val.csv')
def test_cond8_csv_covers_all_eight_ids():
    df = pd.read_csv(MUSES_COND8_VAL)
    assert sorted(df['condition_id'].unique()) == list(range(8))
