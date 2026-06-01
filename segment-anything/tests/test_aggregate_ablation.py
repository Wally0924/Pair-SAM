"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_aggregate_ablation.py -v
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.aggregate_ablation import mean_std, fmt_cell, load_runs, build_summary_table


def test_mean_std_single_value():
    m, s = mean_std([0.6568])
    assert abs(m - 0.6568) < 1e-9
    assert s == 0.0


def test_mean_std_three_seeds():
    m, s = mean_std([0.64, 0.65, 0.66])
    assert abs(m - 0.65) < 1e-9
    assert s > 0.0


def test_fmt_cell_percent():
    assert fmt_cell(0.6568) == "65.7"
    assert "65.0" in fmt_cell(0.65, 0.01)   # mean±std form contains the mean


def _write_run(root, run_id, seed, overall_miou):
    d = os.path.join(root, f"{run_id}_seed{seed}")
    os.makedirs(d, exist_ok=True)
    cfg = dict(model_type='vit_h', use_vgg_adapter=True, inject='pre',
               decoder='unified', lrh=True, mfb=True, ref=True, seed=seed,
               lovasz_weight=1.0, dice_weight=1.0)
    with open(os.path.join(d, 'ablation_config.json'), 'w') as f:
        json.dump(cfg, f)
    res = dict(
        overall_miou=overall_miou,
        per_condition_miou={'fog': overall_miou+0.02, 'rain': overall_miou,
                            'snow': overall_miou+0.01, 'night': overall_miou-0.05},
        per_class_iou_overall={'rider': 0.30, 'motorcycle': 0.40, 'bicycle': 0.50},
        per_class_iou_by_condition={},
    )
    with open(os.path.join(d, 'e1_results.json'), 'w') as f:
        json.dump(res, f)


def test_load_runs_groups_seeds(tmp_path):
    root = str(tmp_path)
    _write_run(root, 'FULL', 42, 0.65)
    _write_run(root, 'FULL', 1234, 0.66)
    _write_run(root, 'FULL', 2026, 0.64)
    _write_run(root, 'R1', 42, 0.40)
    runs = load_runs(root, results_filename='e1_results.json')
    assert set(runs.keys()) == {'FULL', 'R1'}
    assert len(runs['FULL']['overall_miou']) == 3   # 3 seeds collected
    assert len(runs['R1']['overall_miou']) == 1


def test_build_summary_table_has_rows_and_delta(tmp_path):
    root = str(tmp_path)
    for rid, m in [('R1',0.40),('R2',0.55),('R3',0.58),('R4',0.60),
                   ('R5',0.61),('R6',0.63),('FULL',0.65)]:
        _write_run(root, rid, 42, m)
    runs = load_runs(root, results_filename='e1_results.json')
    tex = build_summary_table(runs)
    assert 'R1' in tex and 'FULL' in tex
    # Δ for R1 ≈ (0.40-0.65)*100 = -25.0
    assert '-25.0' in tex
