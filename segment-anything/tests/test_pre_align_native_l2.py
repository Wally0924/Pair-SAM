"""
TDD — Task 8b: pre_align native l2 resolution.

Tests that CMAAlignment.pre_align(l2_native=True) returns l2 at 2×out_size
(genuine stride-8 resolution) and that the flow field is correctly scaled ×2
when upsampling from out_size to 2×out_size.

Using 256×256 input so VGG16 maxpool pipeline is happy and out_size=(16,16)
keeps CPU runtime short.
"""
import inspect
import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.fusion import CMAAlignment
from segment_anything.modeling.cma_utils import warp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model():
    """Random-init CMAAlignment — shapes only, no pretrained weights."""
    return CMAAlignment(embed_dim=256, pretrained_path=None)


def make_images(H=256, W=256, B=1):
    """Random uint8-range float images in [0, 255]."""
    return (torch.rand(B, 3, H, W) * 255.0,
            torch.rand(B, 3, H, W) * 255.0)


# ---------------------------------------------------------------------------
# Test 1: l2_native=True → shapes are correct
# ---------------------------------------------------------------------------

def test_pre_align_native_l2_shapes():
    """l2 must be 2×out_size, l3 at out_size, mask at 2×out_size."""
    model = make_model()
    img_curr, img_ref = make_images(256, 256)
    out_h, out_w = 16, 16

    out = model.pre_align(img_curr, img_ref, out_size=(out_h, out_w), l2_native=True)

    assert out['l2'].shape[-2:] == (2 * out_h, 2 * out_w), (
        f"Expected l2 spatial {(2*out_h, 2*out_w)}, got {out['l2'].shape[-2:]}"
    )
    assert out['l3'].shape[-2:] == (out_h, out_w), (
        f"Expected l3 spatial {(out_h, out_w)}, got {out['l3'].shape[-2:]}"
    )
    assert out['mask'].shape[-2:] == (2 * out_h, 2 * out_w), (
        f"Expected mask spatial {(2*out_h, 2*out_w)}, got {out['mask'].shape[-2:]}"
    )


# ---------------------------------------------------------------------------
# Test 2: l2_native=True → output is finite (no NaN/Inf)
# ---------------------------------------------------------------------------

def test_pre_align_native_l2_finite():
    """All output tensors must be finite."""
    model = make_model()
    img_curr, img_ref = make_images(256, 256)

    out = model.pre_align(img_curr, img_ref, out_size=(16, 16), l2_native=True)

    assert torch.isfinite(out['l2']).all(), "l2 contains NaN/Inf"
    assert torch.isfinite(out['l3']).all(), "l3 contains NaN/Inf"
    assert torch.isfinite(out['mask']).all(), "mask contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 3: l2_native=False (default) → legacy shapes preserved
# ---------------------------------------------------------------------------

def test_pre_align_legacy_shapes_unchanged():
    """Legacy path (l2_native=False) must still return l2 at out_size."""
    model = make_model()
    img_curr, img_ref = make_images(256, 256)
    out_h, out_w = 16, 16

    out = model.pre_align(img_curr, img_ref, out_size=(out_h, out_w))
    # l2_native defaults to False → old behaviour
    assert out['l2'].shape[-2:] == (out_h, out_w), (
        f"Legacy l2 spatial should be {(out_h, out_w)}, got {out['l2'].shape[-2:]}"
    )
    assert out['l3'].shape[-2:] == (out_h, out_w), (
        f"Legacy l3 spatial should be {(out_h, out_w)}, got {out['l3'].shape[-2:]}"
    )


# ---------------------------------------------------------------------------
# Test 4: ×2 flow scaling is present in source (locked guard)
# ---------------------------------------------------------------------------

def test_flow_x2_scaling_present_in_source():
    """
    The ×2 flow scaling must be present in pre_align source.
    This is a source-level guard that locks the critical correctness property.
    If someone removes '* 2.0', this test breaks loudly.
    """
    src = inspect.getsource(CMAAlignment.pre_align)
    assert '* 2.0' in src, (
        "CRITICAL: '* 2.0' flow scaling is missing from CMAAlignment.pre_align. "
        "Upsampling the flow from out_size to 2×out_size without scaling ×2 "
        "silently halves the alignment displacement (pixel-unit flow convention). "
        "See cma_utils.py::warp: vgrid = pixel_grid + flo."
    )


# ---------------------------------------------------------------------------
# Test 5: Numeric check — ×2 scaling produces the correct pixel shift
#
# Construct a toy scenario:
#   - constant flow field at out_size level pointing rightward by D pixels
#   - at 2×out_size the same physical displacement is 2D pixels
#   - verify the warped feature is shifted by ~2D (not D)
#
# We use the warp() function directly with controlled inputs.
# ---------------------------------------------------------------------------

def test_flow_x2_scaling_correct_numeric():
    """
    Numeric check that upsampling a pixel-unit flow ×2 in size AND ×2 in
    value correctly preserves the physical displacement.

    Setup: feature map of shape (1,1,8,8) where column c has value float(c).
    Flow = D pixels rightward at 4×4 resolution.
    After upsample to 8×8 with ×2 scaling → flow = 2D rightward at 8×8.
    The warped value at column c should equal original value at column (c - 2D).
    """
    D = 1  # rightward shift in out_size (4×4) pixel units
    H_small, W_small = 4, 4
    H_large, W_large = 8, 8

    # Feature map: value at (r, c) = float(c)
    feat = torch.arange(W_large, dtype=torch.float32).view(1, 1, 1, W_large).expand(
        1, 1, H_large, W_large
    ).clone()

    # Constant rightward flow at small resolution: D pixels in x-direction
    flow_small = torch.zeros(1, 2, H_small, W_small)
    flow_small[:, 0, :, :] = float(D)  # x-component = D

    # Upsample with ×2 scaling
    flow_large = F.interpolate(
        flow_small, size=(H_large, W_large), mode='bilinear', align_corners=False
    ) * 2.0
    # flow_large x-component should be 2*D = 2 pixels

    warped, _ = warp(feat, flow_large, return_mask=True)

    # warp() convention: vgrid = pixel_grid + flo
    # output[c] = feat[c + flo_x]  →  with flo_x = 2D,  output[c] = feat[c + 2D]
    expected_shift = 2 * D
    # Check interior columns only (near-boundary cols may be clipped by grid_sample)
    for c in range(0, W_large - expected_shift - 1):
        expected_val = float(c + expected_shift)
        actual_val = warped[0, 0, H_large // 2, c].item()
        assert abs(actual_val - expected_val) < 0.1, (
            f"At col {c}: expected {expected_val:.2f}, got {actual_val:.2f}. "
            f"Flow ×2 scaling may be incorrect (warp convention: output[c]=feat[c+flo_x])."
        )
