# segment-anything/tests/test_rare_class_sampler.py
"""
執行：python -m pytest segment-anything/tests/test_rare_class_sampler.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.rare_class_sampler import RareClassSampler


def _toy():
    # 10 images (idx 0..9); class 0 common (idx 0..7), class 1 mid (idx 8,9), class 2 rare (idx 9)
    presence = [[0]]*8 + [[0, 1], [1, 2]]
    pixel_counts = [800, 20, 2]
    return presence, pixel_counts


def test_class_probs_softmax_formula():
    presence, counts = _toy()
    s = RareClassSampler(presence, counts, num_samples=10, temperature=0.01, seed=42, num_classes=3)
    f = torch.tensor(counts, dtype=torch.float64); f = f / f.sum()
    expected = torch.softmax((1.0 - f) / 0.01, dim=0)
    got = torch.tensor(s.class_probs, dtype=torch.float64)
    assert torch.allclose(got, expected, atol=1e-6)


def test_rare_class_drawn_far_more_than_uniform():
    presence, counts = _toy()
    s = RareClassSampler(presence, counts, num_samples=10, temperature=0.01, seed=42, num_classes=3)
    from collections import Counter
    cc = Counter(s._draw_one_class() for _ in range(30000))
    assert cc[2] / 30000 > 0.30      # rare class oversampled vs uniform 1/3
    assert cc[2] > cc[0]


def test_iter_yields_valid_indices_and_length():
    presence, counts = _toy()
    s = RareClassSampler(presence, counts, num_samples=10, temperature=0.01, seed=42, num_classes=3)
    idxs = list(iter(s))
    assert len(idxs) == 10
    assert all(0 <= i < 10 for i in idxs)
    assert 9 in idxs                 # class 2 only in idx 9


def test_reproducible_with_seed():
    presence, counts = _toy()
    a = list(iter(RareClassSampler(presence, counts, 200, 0.01, seed=123, num_classes=3)))
    b = list(iter(RareClassSampler(presence, counts, 200, 0.01, seed=123, num_classes=3)))
    c = list(iter(RareClassSampler(presence, counts, 200, 0.01, seed=999, num_classes=3)))
    assert a == b
    assert a != c


def test_empty_class_excluded():
    presence = [[0], [0], [2], [2]]   # class 1 has no images
    counts = [100, 0, 5]
    s = RareClassSampler(presence, counts, num_samples=100, temperature=0.01, seed=7, num_classes=3)
    assert s.class_probs[1] == 0.0
    assert 1 not in [s._draw_one_class() for _ in range(2000)]
