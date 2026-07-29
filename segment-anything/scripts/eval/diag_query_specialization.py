# segment-anything/scripts/eval/diag_query_specialization.py
"""
零訓練診斷：驗證「類別專屬查詢」的固定對應是否真的成立。

動機
----
Pair-SAM 以損失的索引指派（labels[c]=c）把查詢 q_k 綁定類別 k，但這是**訓練逼出來的
統計性專一化，不是架構鎖死**。本腳本用現成 checkpoint、純推論 + 後處理（不訓練），
量化這個綁定在驗證集上到底有多可靠。

量測兩件事
----------
1) 專一化程度：對每個查詢 q_k，看其分類頭在 C+1=20 維上的 argmax 落點——
     == k            → 預測「自身類別存在」
     == 19 (no-object)→ 預測「自身類別缺席」
     == j≠k (其他類)  → **串台 (cross-talk)**，固定對應在該樣本失效
   若串台率 ≈ 0%，代表綁定其實近乎硬性。

2) 出席判斷準確率：以每張影像的 GT 類別出席為真值，把「argmax==k」視為
   「查詢 q_k 判定自身類別存在」，計算每個查詢的 precision / recall / F1。
   recall 低 = 該類在場卻常被漏（勾了 no-object 或串台）；
   precision 低 = 該類缺席卻常被幻覺為存在。

另按 4 種天氣條件（fog/rain/snow/night）拆解串台率與 recall，
用以檢視退化條件是否讓固定對應鬆動——對應天氣魯棒性論述。

用法
----
    conda run -n sam_env python scripts/eval/diag_query_specialization.py \
        --ckpt outputs_xxx/best_model.pth \
        [--config outputs_xxx/ablation_config.json] \
        [--csv /path/to/val.csv] [--out docs/experiments/diag_query.json]
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (  # noqa: E402
    load_pair_sam_from_ablation,
    build_acdc_val_loader, make_batched_input,
    CONDITION_NAMES, CITYSCAPES_CLASSES, OUTPUT_ROOT,
)

NUM_CLASSES = 19
NO_OBJECT = NUM_CLASSES          # 分類頭第 20 維（index 19）為 no-object
IGNORE_INDEX = 255
DEVICE = 'cuda'


def gt_presence(gt_mask: torch.Tensor, invalid: torch.Tensor) -> np.ndarray:
    """回傳長度 19 的 bool 向量：該類是否在這張影像的有效 GT 像素中出現。"""
    gt = gt_mask.clone()
    gt[invalid] = IGNORE_INDEX
    ids = torch.unique(gt)
    present = np.zeros(NUM_CLASSES, dtype=bool)
    for v in ids.tolist():
        if 0 <= v < NUM_CLASSES:
            present[v] = True
    return present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='19-fixed run 的 checkpoint 路徑')
    ap.add_argument('--config', default=None, help='ablation_config.json（預設取 ckpt 同目錄）')
    ap.add_argument('--csv', default=None, help='val CSV（預設 ACDC val）')
    ap.add_argument('--out', default=None, help='輸出 JSON 路徑')
    ap.add_argument('--name', default='ACDC val', help='資料集名稱，僅用於標題')
    args = ap.parse_args()

    model, cfg = load_pair_sam_from_ablation(args.ckpt, args.config, device=DEVICE)
    print(f"[diag] loaded run config: {cfg}")
    loader = build_acdc_val_loader(args.csv) if args.csv else build_acdc_val_loader()

    # ── 累積器 ──
    # decision 落點計數（每個查詢 k）
    n_own   = np.zeros(NUM_CLASSES, dtype=np.int64)   # argmax == k
    n_noobj = np.zeros(NUM_CLASSES, dtype=np.int64)   # argmax == no-object
    n_cross = np.zeros(NUM_CLASSES, dtype=np.int64)   # argmax == j≠k（串台）
    leak    = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # k 串到 j 的次數
    # 出席判斷混淆（把 argmax==k 當作「預測 k 存在」）
    TP = np.zeros(NUM_CLASSES, dtype=np.int64)
    FP = np.zeros(NUM_CLASSES, dtype=np.int64)
    FN = np.zeros(NUM_CLASSES, dtype=np.int64)
    TN = np.zeros(NUM_CLASSES, dtype=np.int64)
    # 按天氣條件拆解
    cond_decisions = {cid: 0 for cid in CONDITION_NAMES}
    cond_cross     = {cid: 0 for cid in CONDITION_NAMES}
    cond_present   = {cid: 0 for cid in CONDITION_NAMES}   # GT 在場的 (影像,類別) 數
    cond_hit       = {cid: 0 for cid in CONDITION_NAMES}   # 其中 argmax==k 命中數
    n_images = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='diag query specialization'):
            batched_input = make_batched_input(batch, DEVICE)
            outputs = model(batched_input)

            pl = outputs[0]['pred_logits']
            if pl.dim() == 3:            # (1, 19, 20) → (19, 20)
                pl = pl[0]
            argmax = pl.argmax(dim=-1).cpu().numpy()        # (19,) 每個查詢的落點

            gt_mask = batch['gt_mask'][0].to(DEVICE).long()
            invalid = batch['invalid_mask'][0].to(DEVICE)
            cid     = int(batch['condition_id'][0].item())
            present = gt_presence(gt_mask, invalid)          # (19,) bool
            n_images += 1

            for k in range(NUM_CLASSES):
                a = int(argmax[k])
                # 落點分類
                if a == k:
                    n_own[k] += 1
                elif a == NO_OBJECT:
                    n_noobj[k] += 1
                else:
                    n_cross[k] += 1
                    leak[k, a] += 1
                    cond_cross[cid] += 1
                cond_decisions[cid] += 1

                # 出席判斷混淆（pred_present := argmax==k）
                pred_present = (a == k)
                if pred_present and present[k]:
                    TP[k] += 1
                elif pred_present and not present[k]:
                    FP[k] += 1
                elif (not pred_present) and present[k]:
                    FN[k] += 1
                else:
                    TN[k] += 1

                if present[k]:
                    cond_present[cid] += 1
                    if pred_present:
                        cond_hit[cid] += 1

    # ── 指標 ──
    def safe_div(a, b):
        return float(a) / float(b) if b else float('nan')

    decisions_per_q = n_own + n_noobj + n_cross            # == n_images（每查詢每圖一次）
    crosstalk_rate = n_cross / np.maximum(decisions_per_q, 1)   # 每查詢串台率
    precision = np.array([safe_div(TP[k], TP[k] + FP[k]) for k in range(NUM_CLASSES)])
    recall    = np.array([safe_div(TP[k], TP[k] + FN[k]) for k in range(NUM_CLASSES)])
    f1 = np.array([
        safe_div(2 * precision[k] * recall[k], precision[k] + recall[k])
        if not (math.isnan(precision[k]) or math.isnan(recall[k])) and (precision[k] + recall[k]) > 0
        else float('nan')
        for k in range(NUM_CLASSES)
    ])

    total_decisions = int(decisions_per_q.sum())
    total_cross = int(n_cross.sum())
    overall_specialization = 1.0 - safe_div(total_cross, total_decisions)   # argmax ∈ {k, no-object} 的比例

    def macro(arr):
        vals = [v for v in arr if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    # ── 組 JSON ──
    per_query = {}
    for k in range(NUM_CLASSES):
        per_query[CITYSCAPES_CLASSES[k]] = {
            'gt_present_images': int(TP[k] + FN[k]),
            'argmax_own_pct':      100.0 * safe_div(n_own[k], decisions_per_q[k]),
            'argmax_noobject_pct': 100.0 * safe_div(n_noobj[k], decisions_per_q[k]),
            'crosstalk_pct':       100.0 * crosstalk_rate[k],
            'presence_precision':  None if math.isnan(precision[k]) else round(float(precision[k]), 4),
            'presence_recall':     None if math.isnan(recall[k]) else round(float(recall[k]), 4),
            'presence_f1':         None if math.isnan(f1[k]) else round(float(f1[k]), 4),
        }
    per_condition = {}
    for cid in CONDITION_NAMES:
        per_condition[CONDITION_NAMES[cid]] = {
            'crosstalk_pct':  100.0 * safe_div(cond_cross[cid], cond_decisions[cid]),
            'presence_recall': safe_div(cond_hit[cid], cond_present[cid]),
        }
    json_data = {
        'checkpoint': str(Path(args.ckpt).name),
        'dataset': args.name,
        'num_images': n_images,
        'overall_specialization_rate': round(overall_specialization, 5),
        'overall_crosstalk_pct': round(100.0 * safe_div(total_cross, total_decisions), 5),
        'macro_presence_precision': round(macro(precision), 4),
        'macro_presence_recall': round(macro(recall), 4),
        'macro_presence_f1': round(macro(f1), 4),
        'per_query': per_query,
        'per_condition': per_condition,
    }

    if args.out is not None:
        json_path = Path(args.out)
    else:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        json_path = OUTPUT_ROOT / 'diag_query_specialization.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f'✅ JSON written: {json_path}')

    # ── Markdown ──
    md = []
    md.append(f'# 查詢專一化診斷 — {args.name}')
    md.append('')
    md.append(f'**Checkpoint:** `{Path(args.ckpt).name}`　**Date:** {datetime.now().strftime("%Y-%m-%d")}　**Images:** {n_images}')
    md.append('')
    md.append(f'- **整體專一化率**（argmax ∈ {{自身類別, no-object}}）：**{overall_specialization*100:.2f}%**')
    md.append(f'- **整體串台率**（argmax 落到其他類別）：**{100.0*safe_div(total_cross, total_decisions):.3f}%**')
    md.append(f'- **Macro 出席 precision / recall / F1**：'
              f'{macro(precision)*100:.1f}% / {macro(recall)*100:.1f}% / {macro(f1)*100:.1f}%')
    md.append('')
    md.append('## Per-Query')
    md.append('')
    md.append('| Class (q_k) | GT在場圖數 | argmax=自身% | argmax=no-obj% | 串台% | 出席P | 出席R | F1 |')
    md.append('|---|--:|--:|--:|--:|--:|--:|--:|')
    for k in range(NUM_CLASSES):
        q = per_query[CITYSCAPES_CLASSES[k]]
        def pct(v): return '—' if v is None else f'{v*100:.1f}'
        md.append('| {} | {} | {:.1f} | {:.1f} | {:.2f} | {} | {} | {} |'.format(
            CITYSCAPES_CLASSES[k], q['gt_present_images'],
            q['argmax_own_pct'], q['argmax_noobject_pct'], q['crosstalk_pct'],
            pct(q['presence_precision']), pct(q['presence_recall']), pct(q['presence_f1']),
        ))
    md.append('')
    md.append('## Per-Condition')
    md.append('')
    md.append('| Condition | 串台% | 出席 recall% |')
    md.append('|---|--:|--:|')
    for cid in CONDITION_NAMES:
        pc = per_condition[CONDITION_NAMES[cid]]
        rec = pc['presence_recall']
        md.append(f'| {CONDITION_NAMES[cid].capitalize()} | {pc["crosstalk_pct"]:.2f} | '
                  f'{"—" if math.isnan(rec) else f"{rec*100:.1f}"} |')
    md.append('')
    # 串台流向（只列有發生的）
    leaks = [(k, j, int(leak[k, j])) for k in range(NUM_CLASSES) for j in range(NUM_CLASSES) if leak[k, j] > 0]
    if leaks:
        md.append('## 串台流向（q_k 的 argmax 落到別類的次數，僅列 >0）')
        md.append('')
        md.append('| q_k | → 誤判成 | 次數 |')
        md.append('|---|---|--:|')
        for k, j, n in sorted(leaks, key=lambda x: -x[2]):
            md.append(f'| {CITYSCAPES_CLASSES[k]} | {CITYSCAPES_CLASSES[j]} | {n} |')
    else:
        md.append('## 串台流向')
        md.append('')
        md.append('**零串台**：所有查詢的 argmax 都落在 {自身類別, no-object}，固定對應在此驗證集上等同硬性綁定。')
    md.append('')

    md_path = json_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'✅ Markdown written: {md_path}')

    # ── Console 摘要 ──
    print(f'\n=== 摘要（{args.name}, {n_images} 張）===')
    print(f'整體專一化率：{overall_specialization*100:.2f}%　串台率：{100.0*safe_div(total_cross, total_decisions):.3f}%')
    print(f'Macro 出席 P/R/F1：{macro(precision)*100:.1f}% / {macro(recall)*100:.1f}% / {macro(f1)*100:.1f}%')
    worst_recall = sorted(
        [(CITYSCAPES_CLASSES[k], recall[k]) for k in range(NUM_CLASSES) if not math.isnan(recall[k])],
        key=lambda x: x[1])[:5]
    print('出席 recall 最低的 5 類（惡劣天氣最易漏）：',
          ', '.join(f'{c}={r*100:.0f}%' for c, r in worst_recall))


if __name__ == '__main__':
    main()
