# segment-anything/tests/test_semantic_assembly.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_semantic_assembly.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits


def test_scatter_places_classes_and_fills_rest():
    low_res = torch.randn(2, 4, 4)          # K=2, 4x4
    class_ids = [3, 7]
    out = assemble_semantic_logits(low_res, class_ids, fusion_head=None,
                                   num_classes=19, use_lrh=False, fill_value=-10.0)
    assert out.shape == (1, 19, 4, 4)
    assert torch.allclose(out[0, 3], low_res[0])
    assert torch.allclose(out[0, 7], low_res[1])
    assert torch.allclose(out[0, 0], torch.full((4, 4), -10.0))


def test_use_lrh_false_skips_fusion_head():
    low_res = torch.randn(1, 4, 4)
    head = nn.Identity()
    out_off = assemble_semantic_logits(low_res, [0], fusion_head=head,
                                       num_classes=19, use_lrh=False)
    out_raw = assemble_semantic_logits(low_res, [0], fusion_head=None,
                                       num_classes=19, use_lrh=False)
    assert torch.allclose(out_off, out_raw)


def test_use_lrh_true_applies_fusion_head():
    low_res = torch.randn(1, 4, 4)

    class AddOne(nn.Module):
        def forward(self, x):
            return x + 1.0

    out_on = assemble_semantic_logits(low_res, [0], fusion_head=AddOne(),
                                      num_classes=19, use_lrh=True)
    out_raw = assemble_semantic_logits(low_res, [0], fusion_head=None,
                                       num_classes=19, use_lrh=False)
    assert torch.allclose(out_on, out_raw + 1.0)
