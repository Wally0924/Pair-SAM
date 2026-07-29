# tests/test_ablation_flags_w3_w6.py
"""W3(--no-conf_mod)/W6(--no-extractor) 消融旗標的語義鎖定。

W3:移除置信度調變 = use_reference 保持 True、conf 一律中性值 1(m̄≡1,
   參考特徵不分可靠與否全幅注入)。
W6:移除抽取器 = build 於 enable_vgg_adapter 前清空 EXTRACT_BLOCKS/extractors,
   單向注入、參考先驗不隨主幹深度更新,且不得殘留 trainable-but-unused 參數。
"""
import torch
import torch.nn as nn

from segment_anything.modeling.deform_adapter import ReferencePriorModule, DeformAdapter


def _fake_feats(B=1, dim_l2=8, dim_l3=8, dim_l4=8):
    return {
        'l2': torch.randn(B, dim_l2, 8, 8),
        'l3': torch.randn(B, dim_l3, 4, 4),
        'l4': torch.randn(B, dim_l4, 2, 2),
        'mask': torch.rand(B, 1, 16, 16),   # 連續置信度 [0,1)
    }


def _small_rpm(**kw):
    return ReferencePriorModule(l2_channels=8, l3_channels=8, l4_channels=8,
                                dim=16, **kw)


def test_conf_mod_default_uses_mask():
    """預設(use_conf_mod=True):conf 取自參考遮罩置信度,不應全為 1。"""
    torch.manual_seed(0)
    rpm = _small_rpm()
    assert rpm.use_conf_mod is True
    c, conf = rpm(_fake_feats())
    assert not torch.allclose(conf, torch.ones_like(conf)), \
        "預設應為置信度調變,conf 不應恆等於 1"


def test_no_conf_mod_forces_neutral_conf_keeps_reference():
    """W3:use_conf_mod=False → conf≡1,但參考特徵 c 保持非零(仍引入參考)。"""
    torch.manual_seed(0)
    rpm = _small_rpm()
    rpm.use_conf_mod = False        # 模擬 build_pair_sam_from_config 覆蓋
    c, conf = rpm(_fake_feats())
    assert torch.equal(conf, torch.ones_like(conf)), "W3 語義:m̄≡1"
    assert c.abs().sum() > 0, "W3 不得歸零參考特徵(那是 --no-ref 的語義)"


def test_no_ref_still_neutral_conf():
    """語義 B 回歸鎖定:use_reference=False → c=0 且 conf≡1(commit ccc99d9)。"""
    torch.manual_seed(0)
    rpm = _small_rpm(use_reference=False)
    c, conf = rpm(_fake_feats())
    assert torch.equal(c, torch.zeros_like(c))
    assert torch.equal(conf, torch.ones_like(conf))


def test_w6_extractor_removal_invariants():
    """W6:清空後 EXTRACT_BLOCKS=[]、extractors 無參數、set_features 狀態一致。"""
    ad = DeformAdapter(vit_dim=16, l2_channels=8, l3_channels=8, l4_channels=8,
                       n_heads=2)
    # 模擬 build_pair_sam_from_config 的 W6 覆蓋(enable 前執行)
    ad.EXTRACT_BLOCKS = []
    ad.extractors = nn.ModuleList()
    ad._extract_c = []
    assert len(list(ad.extractors.parameters())) == 0, \
        "extractors 需無參數,否則 trainable-but-unused 觸發梯度稽核"
    ad.set_features(_fake_feats(), h=4, w=4)
    assert ad._extract_c == [], "set_features 後 _extract_c 應維持空(依實例 EXTRACT_BLOCKS)"
    assert ad._c is not None, "注入端狀態不受 W6 影響"
