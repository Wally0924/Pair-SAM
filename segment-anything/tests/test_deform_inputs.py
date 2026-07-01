import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import get_reference_points, deform_inputs


def test_reference_points_shape_and_range():
    ref = get_reference_points([(4, 4), (2, 2)], torch.device('cpu'))
    assert ref.shape == (1, 16 + 4, 1, 2)
    assert ref.min() >= 0.0 and ref.max() <= 1.0


def test_deform_inputs_inject_and_extract():
    h = w = 64  # ViT token grid = 1/16 of 1024
    inj, ext = deform_inputs(h, w, torch.device('cpu'))
    # inject: value 有 3 尺度 (h*2)²,(h)²,(h/2)²
    assert inj[1].tolist() == [[128, 128], [64, 64], [32, 32]]
    assert inj[0].shape == (1, 64 * 64, 1, 2)          # query = 1/16 grid
    assert inj[2].tolist() == [0, 128 * 128, 128 * 128 + 64 * 64]
    # extract: value 為單一 1/16，query = 3 尺度
    assert ext[1].tolist() == [[64, 64]]
    assert ext[0].shape == (1, 128 * 128 + 64 * 64 + 32 * 32, 1, 2)
    assert ext[2].tolist() == [0]
