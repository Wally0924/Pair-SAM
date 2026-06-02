# segment-anything/tests/test_precompute_class_presence.py
"""
執行：python -m pytest segment-anything/tests/test_precompute_class_presence.py -v
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from scripts.precompute_class_presence import scan_gt_mask, build_class_presence


def test_scan_gt_mask_excludes_255(tmp_path):
    m = np.full((8, 8), 255, dtype=np.uint8)
    m[0:4, 0:4] = 3      # class 3
    m[4:8, 4:8] = 7      # class 7
    p = str(tmp_path / "gt.png")
    cv2.imwrite(p, m)
    present, counts = scan_gt_mask(p, num_classes=19)
    assert set(present) == {3, 7}
    assert counts[3] == 16 and counts[7] == 16
    assert counts[0] == 0


def test_build_class_presence_over_csv(tmp_path):
    a = np.zeros((4, 4), dtype=np.uint8); a[0, 0] = 18   # A: {0,18}
    b = np.zeros((4, 4), dtype=np.uint8)                 # B: {0}
    pa, pb = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    cv2.imwrite(pa, a); cv2.imwrite(pb, b)
    csv_p = str(tmp_path / "train.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gt_path"]); w.writerow([pa]); w.writerow([pb])
    out = str(tmp_path / "cp.json")
    data = build_class_presence(csv_p, out, num_classes=19)
    assert set(data["presence"][pa]) == {0, 18}
    assert set(data["presence"][pb]) == {0}
    assert data["class_pixel_counts"][18] == 1
    assert data["class_pixel_counts"][0] == 31   # 15 (a) + 16 (b)
    assert os.path.isfile(out)
    with open(out) as f:
        loaded = json.load(f)
    # JSON keys 可能為字串；容忍兩種
    cpc = loaded["class_pixel_counts"]
    assert cpc.get("18", cpc.get(18)) == 1
