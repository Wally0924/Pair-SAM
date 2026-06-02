"""
aggregate_ablation.py — 掃描 ablation run 目錄，輸出三張消融表的 LaTeX tabular body。

使用方式：
    conda run -n sam_env python segment-anything/scripts/aggregate_ablation.py \
        --runs_root outputs_ablation \
        --out ablation_tables.tex

產生的檔案包含三個區塊，以 % tab:<name> 標記分隔，可直接貼入 .tex skeleton。
"""

import argparse
import json
import math
import os


# ---------------------------------------------------------------------------
# 純函式工具
# ---------------------------------------------------------------------------

def mean_std(values):
    """計算平均值與樣本標準差（n<2 → std=0.0）。

    Args:
        values: 數值列表（float），必須非空。

    Returns:
        (mean, std) tuple，均為 float。
    """
    n = len(values)
    if n == 0:
        raise ValueError("mean_std requires at least one value")
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(variance)


def fmt_cell(mean, std=0.0):
    """將 [0,1] 區間的 mIoU 格式化為百分比字串（×100，一位小數）。

    若 std > 0，輸出 "XX.X$\\pm$Y.Y"；否則輸出 "XX.X"。

    Args:
        mean: float，均值（[0,1] 區間）。
        std:  float，標準差（[0,1] 區間），預設 0.0。

    Returns:
        LaTeX 格式字串。
    """
    m_pct = mean * 100
    if std is not None and std * 100 >= 0.05:
        s_pct = std * 100
        return f"{m_pct:.1f}$\\pm${s_pct:.1f}"
    return f"{m_pct:.1f}"


# ---------------------------------------------------------------------------
# 資料載入
# ---------------------------------------------------------------------------

def load_runs(runs_root, results_filename='e1_results.json'):
    """掃描 runs_root 下所有子目錄，依 RunID 分組彙整結果。

    目錄命名規則：<RunID>_seed<N>（如 R1_seed42、FULL_seed1234）。
    同一 RunID 的多個 seed 結果被收集到列表中，供後續計算 mean±std。

    Args:
        runs_root: str，根目錄路徑。
        results_filename: str，每個 run 目錄內的結果 JSON 檔名。

    Returns:
        dict，key = RunID，value = {
            'overall_miou': [float, ...],
            'per_condition_miou': [dict, ...],  # 每個 seed 的條件 mIoU dict
            'per_class_iou_overall': [dict, ...],
            'cfg': dict,  # 最後一個讀到的 ablation_config（同 RunID 應相同）
        }
    """
    runs = {}
    if not os.path.isdir(runs_root):
        return runs

    for entry in sorted(os.listdir(runs_root)):
        entry_path = os.path.join(runs_root, entry)
        if not os.path.isdir(entry_path):
            continue

        # 解析 RunID：取 _seed 之前的部分
        if '_seed' not in entry:
            continue
        run_id, _ = entry.rsplit('_seed', 1)

        cfg_path = os.path.join(entry_path, 'ablation_config.json')
        res_path = os.path.join(entry_path, results_filename)
        if not os.path.isfile(cfg_path) or not os.path.isfile(res_path):
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        with open(res_path) as f:
            res = json.load(f)

        if run_id not in runs:
            runs[run_id] = {
                'overall_miou': [],
                'per_condition_miou': [],
                'per_class_iou_overall': [],
                'cfg': cfg,
            }

        runs[run_id]['overall_miou'].append(res['overall_miou'])
        runs[run_id]['per_condition_miou'].append(res.get('per_condition_miou', {}))
        runs[run_id]['per_class_iou_overall'].append(res.get('per_class_iou_overall', {}))
        runs[run_id]['cfg'] = cfg  # 最後一個（同 RunID 應相同）

    return runs


# ---------------------------------------------------------------------------
# 輔助：聚合單一 RunID 的指標
# ---------------------------------------------------------------------------

def _agg_overall(runs, run_id):
    """回傳 (mean, std) 的 overall mIoU（[0,1] 區間）。"""
    return mean_std(runs[run_id]['overall_miou'])


def _agg_condition(runs, run_id, cond):
    """回傳 (mean, std) 的指定條件 mIoU（[0,1] 區間）。"""
    vals = [d.get(cond, float('nan')) for d in runs[run_id]['per_condition_miou']]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float('nan'), 0.0
    return mean_std(vals)


def _agg_class(runs, run_id, cls_name):
    """回傳 (mean, std) 的指定類別 IoU（[0,1] 區間）。null → 跳過。"""
    vals = []
    for d in runs[run_id]['per_class_iou_overall']:
        v = d.get(cls_name)
        if v is not None:
            vals.append(v)
    if not vals:
        return float('nan'), 0.0
    return mean_std(vals)


def _delta_str(row_mean, full_mean):
    """格式化 Δ = (row_mean − full_mean)×100，帶正負號，一位小數。"""
    d = (row_mean - full_mean) * 100
    return f"{d:+.1f}"


# ---------------------------------------------------------------------------
# 表格建構器
# ---------------------------------------------------------------------------

def build_summary_table(runs):
    """建構 tab:ablation_summary 的 LaTeX tabular body。

    列順序：R1, R2, R3, R4, R5, R6, R7, R8（FULL = R8，+RCS 行）
    欄位：All mIoU (%), Δ (vs FULL=R8, pp)
    R1 & R8 顯示 mean±std（若有多個 seed）；其他顯示 mean。

    Args:
        runs: load_runs() 的回傳值。

    Returns:
        str，LaTeX 資料列（每列以 \\\\ 結尾）。
    """
    row_order = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    missing = [r for r in row_order if r not in runs]
    if missing:
        return f"% tab:ablation_summary: missing runs {', '.join(missing)}\n"

    full_mean, _ = _agg_overall(runs, 'R8')
    # R1 & R8(FULL) 顯示 mean±std；其餘只顯示 mean
    multi_seed_rows = {'R1', 'R8'}

    lines = []
    for rid in row_order:
        m, s = _agg_overall(runs, rid)
        use_std = rid in multi_seed_rows and s > 0.0
        cell = fmt_cell(m, s if use_std else 0.0)
        delta = "---" if rid == 'R8' else _delta_str(m, full_mean)
        lines.append(f"{rid} & {cell} & {delta} \\\\")

    return "\n".join(lines) + "\n"


def build_adapter_table(runs):
    """建構 tab:adapter_ablation 的 LaTeX tabular body。

    列順序：FULL(=R8), A1, A2
    欄位：Fog, Rain, Snow, Night, All (%), Δ (vs FULL=R8)
    FULL(R8) & A2 顯示 mean±std（若有多個 seed）。

    Args:
        runs: load_runs() 的回傳值。

    Returns:
        str，LaTeX 資料列。
    """
    # 列定義：(顯示名稱, 實際 RunID, 是否顯示 std)
    row_defs = [
        ('FULL', 'R8', True),
        ('A1',   'A1', False),
        ('A2',   'A2', True),
    ]
    required = [rid for _, rid, _ in row_defs]
    missing = [r for r in required if r not in runs]
    if missing:
        missing_display = ['FULL(R8)' if r == 'R8' else r for r in missing]
        return f"% tab:adapter_ablation: missing runs {', '.join(missing_display)}\n"

    full_mean, _ = _agg_overall(runs, 'R8')
    conditions = ['fog', 'rain', 'snow', 'night']

    lines = []
    for display_id, rid, use_std_flag in row_defs:
        overall_m, overall_s = _agg_overall(runs, rid)
        use_std = use_std_flag and overall_s > 0.0

        cond_cells = []
        for cond in conditions:
            cm, cs = _agg_condition(runs, rid, cond)
            use_cond_std = use_std_flag and cs > 0.0
            cond_cells.append(fmt_cell(cm, cs if use_cond_std else 0.0))

        all_cell = fmt_cell(overall_m, overall_s if use_std else 0.0)
        delta = "---" if display_id == 'FULL' else _delta_str(overall_m, full_mean)
        row = " & ".join([display_id] + cond_cells + [all_cell, delta]) + " \\\\"
        lines.append(row)

    return "\n".join(lines) + "\n"


def build_loss_table(runs):
    """建構 tab:loss_ablation 的 LaTeX tabular body。

    列順序：FULL(=R8), C1, C2
    C2 讀取自己的 run dir（C2_seed*），不再重用 R6。
    欄位：All mIoU (%), rider IoU, motorcycle IoU, bicycle IoU, Δ All
    FULL(R8) 顯示 mean±std（若有多個 seed）。

    Args:
        runs: load_runs() 的回傳值。

    Returns:
        str，LaTeX 資料列。
    """
    # 列定義：(顯示名稱, 實際 RunID, 是否顯示 std)
    row_defs = [
        ('FULL', 'R8', True),
        ('C1',   'C1', False),
        ('C2',   'C2', False),
    ]
    required = [rid for _, rid, _ in row_defs]
    missing = [r for r in required if r not in runs]
    if missing:
        missing_display = ['FULL(R8)' if r == 'R8' else r for r in missing]
        return f"% tab:loss_ablation: missing runs {', '.join(missing_display)}\n"

    full_mean, _ = _agg_overall(runs, 'R8')

    lines = []
    for display_id, rid, use_std_flag in row_defs:
        overall_m, overall_s = _agg_overall(runs, rid)
        use_std = use_std_flag and overall_s > 0.0
        all_cell = fmt_cell(overall_m, overall_s if use_std else 0.0)

        class_cells = []
        for cls_name in ['rider', 'motorcycle', 'bicycle']:
            cm, cs = _agg_class(runs, rid, cls_name)
            use_cls_std = use_std_flag and cs > 0.0
            if math.isnan(cm):
                class_cells.append('--')
            else:
                class_cells.append(fmt_cell(cm, cs if use_cls_std else 0.0))

        delta = "---" if display_id == 'FULL' else _delta_str(overall_m, full_mean)
        row = " & ".join([display_id, all_cell] + class_cells + [delta]) + " \\\\"
        lines.append(row)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI 主程式
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='掃描 ablation run 目錄，輸出三張消融表的 LaTeX tabular body。'
    )
    parser.add_argument('--runs_root', required=True,
                        help='包含所有 run 子目錄的根目錄路徑。')
    parser.add_argument('--results_filename', default='e1_results.json',
                        help='每個 run 目錄內的結果 JSON 檔名（預設：e1_results.json）。')
    parser.add_argument('--out', default='ablation_tables.tex',
                        help='輸出 LaTeX 檔案路徑（預設：ablation_tables.tex）。')
    args = parser.parse_args()

    runs = load_runs(args.runs_root, results_filename=args.results_filename)

    found_ids = sorted(runs.keys())
    print(f"[aggregate_ablation] 找到 {len(runs)} 個 RunID：{found_ids}")
    for rid, data in runs.items():
        n = len(data['overall_miou'])
        m, s = mean_std(data['overall_miou'])
        print(f"  {rid:12s}  seeds={n}  mIoU={fmt_cell(m, s if n > 1 else 0.0)}%")

    summary_body  = build_summary_table(runs)
    adapter_body  = build_adapter_table(runs)
    loss_body     = build_loss_table(runs)

    output = (
        "% tab:ablation_summary\n"
        + summary_body
        + "\n% tab:adapter_ablation\n"
        + adapter_body
        + "\n% tab:loss_ablation\n"
        + loss_body
    )

    with open(args.out, 'w') as f:
        f.write(output)

    print(f"\n[aggregate_ablation] 輸出已寫入 {args.out}")
    print("--- 預覽 ---")
    print(output)


if __name__ == '__main__':
    main()
