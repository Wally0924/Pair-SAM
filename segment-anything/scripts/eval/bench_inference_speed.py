#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""實驗 C：Pair-SAM 推論效率量測（口試 Q15）。

量測項目（單卡、batch_size=1、1024x1024 輸入）：
  - 端到端單張延遲：中位數 / p90 / 平均±標準差
  - 峰值 VRAM
  - 參數量（總計 / 可訓練），與論文表 tab:trainable_params 交叉驗證

量測方法：
  - torch.cuda.synchronize() 前後夾住，避免非同步造成低估
  - 充分 warmup（預設 10 次）後才計時，排除 cudnn autotune 與快取效應
  - 回報中位數而非平均，降低偶發排程抖動影響
  - torch.inference_mode() + AMP 設定與 eval_e1_acdc_val_full.py 一致

對照設定：
  FULL     完整模型（含參考影像與對齊網路）
  W2_semB  無參考影像（參考先驗置零、置信度設為 1，Adapter 結構不變）
  B0       無 Adapter 基線

注意：FULL 與 W2_semB 兩者的 forward 都會執行 pre_align（VGG 特徵萃取 +
UAWarpC 對齊）——_adapter_reference_free 僅在 adapter_variant='sam_adapter'
時為 True，W2_semB 的 adapter_variant 是 'reference'，不符合此條件。因此
FULL − W2_semB 的延遲差只反映 RPM 之後參考特徵是否歸零，不代表對齊網路的
成本。若要估計對齊網路本身的成本，應改用 --bench-pre-align 直接對
fusion_module.pre_align() 計時，或改以 W4（adapter_variant='sam_adapter'，
真正跳過 pre_align）與 FULL 的延遲差估計。

用法：
    conda run -n sam_env python scripts/eval/bench_inference_speed.py
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.eval._eval_common import (  # noqa: E402
    load_pair_sam_from_ablation, build_acdc_val_loader, make_batched_input,
)

ABL = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs_ablation_m2f')
RUNS = [
    ('FULL (完整模型)', 'FULL_seed42'),
    ('無參考影像', 'W2_semB_seed42'),
    ('無 Adapter 基線', 'B0_seed42'),
]


def count_params(model):
    """推論階段的總參數量。

    注意：此處不回報「可訓練參數」——模型以 eval 模式載入，requires_grad
    狀態不反映訓練時的凍結設定。訓練階段的可訓練參數分解見論文
    表 tab:trainable_params（40.20 M，佔 4.75%）。
    """
    return sum(p.numel() for p in model.parameters())


def bench(model, batches, warmup, iters, device='cuda'):
    """回傳 (latencies_ms, peak_mem_MB)。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        # warmup：讓 cudnn 選好演算法、配置好快取
        for i in range(warmup):
            model(batches[i % len(batches)])
        torch.cuda.synchronize(device)

        lat = []
        for i in range(iters):
            bi = batches[i % len(batches)]
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(bi)
            torch.cuda.synchronize(device)   # 必須，否則量到的是 kernel launch 時間
            lat.append((time.perf_counter() - t0) * 1000.0)

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return lat, peak


def bench_pre_align(model, batches, warmup, iters, device='cuda'):
    """獨立量測 fusion_module.pre_align()（VGG 特徵萃取 + UAWarpC 對齊）本身的延遲。

    這是對齊網路成本的直接量測，取代原本用 FULL − W2_semB 差值推估的作法——
    兩者其實都會執行 pre_align，差值並不代表對齊網路成本。呼叫方式與
    pair_sam.py forward 階段 0 一致（見該檔案 pre_align 呼叫處）。
    回傳 (latencies_ms, peak_mem_MB)。
    """
    grid = model.image_encoder.img_size // model.image_encoder.patch_embed.proj.stride[0]

    def _stacked(bi):
        img_curr = torch.stack([x['image'] for x in bi], dim=0).to(device)
        img_ref = torch.stack([x['clear_image'] for x in bi], dim=0).to(device)
        return img_curr, img_ref

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for i in range(warmup):
            img_curr, img_ref = _stacked(batches[i % len(batches)])
            model.fusion_module.pre_align(img_curr, img_ref, out_size=(grid, grid), l2_native=True)
        torch.cuda.synchronize(device)

        lat = []
        for i in range(iters):
            img_curr, img_ref = _stacked(batches[i % len(batches)])
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model.fusion_module.pre_align(img_curr, img_ref, out_size=(grid, grid), l2_native=True)
            torch.cuda.synchronize(device)
            lat.append((time.perf_counter() - t0) * 1000.0)

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return lat, peak


def run_once(label, run, ckpt, raw, args, dev):
    """單次完整流程：載入模型 → warmup → bench →釋放。

    每次呼叫都重新 load_pair_sam_from_ablation + 重新 warmup，用於估計
    「重複間」(跨 process/跨 model load) 的變異，而非單一 run 內的排程抖動。
    找不到 checkpoint 時回傳 None。
    """
    if not os.path.exists(ckpt):
        print(f'[跳過] {label}: 找不到 {ckpt}')
        return None
    model, cfg = load_pair_sam_from_ablation(ckpt, device=dev)
    total = count_params(model)

    batches = [make_batched_input(b, dev) for b in raw]
    lat, peak = bench(model, batches, args.warmup, args.iters, dev)

    med = statistics.median(lat)
    row = dict(
        label=label, run=run,
        median_ms=round(med, 2),
        mean_ms=round(statistics.mean(lat), 2),
        std_ms=round(statistics.pstdev(lat), 2),
        p90_ms=round(sorted(lat)[int(len(lat) * 0.9)], 2),
        fps=round(1000.0 / med, 3),
        peak_vram_mb=round(peak, 1),
        params_total_m=round(total / 1e6, 2),
    )
    del model, batches
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--n-batch', type=int, default=5, help='輪替使用的樣本數')
    ap.add_argument('--repeats', type=int, default=1,
                     help='每組設定獨立重複量測次數（各自重新 load model + warmup），'
                          '用以估計重複間變異；>1 時採 FULL/W2_semB/B0 交替量測，'
                          '抵消 GPU 溫度/時脈隨時間漂移的系統性偏差')
    ap.add_argument('--bench-pre-align', action='store_true',
                     help='額外對 FULL 模型的 fusion_module.pre_align() 直接計時，'
                          '作為對齊網路成本的量測（取代不可靠的 FULL−W2_semB 差值推估）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dev = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'torch {torch.__version__}\n')
    print(f'量測設定：warmup={args.warmup}, iters={args.iters}, '
          f'輸入 1024x1024, batch_size=1\n')

    # 預先載入固定樣本，所有設定共用同一批輸入
    loader = build_acdc_val_loader(batch_size=1, num_workers=2)
    raw = []
    for i, b in enumerate(loader):
        if i >= args.n_batch:
            break
        raw.append(b)

    # 交替量測：repeats=1 時等同原本「FULL, W2_semB, B0」單一輪；
    # repeats>1 時以 (rep0: FULL,W2,B0), (rep1: FULL,W2,B0), ... 的順序交替，
    # 讓每組設定的重複都均勻分散在時間軸上，抵消時脈/溫度隨時間漂移的系統性偏差。
    by_label = {label: [] for label, run in RUNS}
    for rep in range(args.repeats):
        for label, run in RUNS:
            ckpt = os.path.join(ABL, run, 'weather_sam_best_latest.pth')
            tag = f'[重複 {rep + 1}/{args.repeats}] ' if args.repeats > 1 else ''
            print(f'── {tag}{label} ({run}) 載入中…', flush=True)
            row = run_once(label, run, ckpt, raw, args, dev)
            if row is None:
                continue
            row['repeat'] = rep + 1
            by_label[label].append(row)
            print(f'   延遲中位數 {row["median_ms"]:.1f} ms  '
                  f'(p90 {row["p90_ms"]:.1f}, 平均 {row["mean_ms"]:.1f}±{row["std_ms"]:.1f})  '
                  f'{row["fps"]:.2f} FPS')
            print(f'   峰值 VRAM {row["peak_vram_mb"]:.0f} MB   '
                  f'參數 {row["params_total_m"]:.1f} M\n')

    # 彙總：repeats=1 時 results 與原本輸出格式一致；
    # repeats>1 時額外附上「重複間」統計量（跨 model load 的變異，非單一 run 內抖動）。
    results = []
    for label, run in RUNS:
        rows = by_label[label]
        if not rows:
            continue
        if args.repeats <= 1:
            row = dict(rows[0])
            row.pop('repeat', None)
            results.append(row)
        else:
            medians = [r['median_ms'] for r in rows]
            repeat_mean = statistics.mean(medians)
            repeat_std = statistics.stdev(medians) if len(medians) > 1 else 0.0
            results.append(dict(
                label=label, run=run,
                n_repeats=len(rows),
                repeat_median_values_ms=medians,
                median_ms=round(repeat_mean, 2),
                repeat_mean_ms=round(repeat_mean, 2),
                repeat_std_ms=round(repeat_std, 2),
                repeat_min_ms=round(min(medians), 2),
                repeat_max_ms=round(max(medians), 2),
                fps=round(1000.0 / repeat_mean, 3),
                peak_vram_mb=round(max(r['peak_vram_mb'] for r in rows), 1),
                params_total_m=rows[0]['params_total_m'],
            ))

    print('=' * 78)
    print(f'{"設定":<18}{"延遲(ms)":>11}{"FPS":>8}{"VRAM(MB)":>11}{"參數(M)":>10}')
    print('-' * 78)
    for r in results:
        print(f'{r["label"]:<18}{r["median_ms"]:>11.1f}{r["fps"]:>8.2f}'
              f'{r["peak_vram_mb"]:>11.0f}{r["params_total_m"]:>10.1f}')
    print('=' * 78)

    if args.repeats > 1:
        print('\n重複間變異（各組獨立重新 load+warmup+bench 的中位數延遲，共 '
              f'{args.repeats} 次）：')
        for r in results:
            print(f'  {r["label"]:<18} 各次中位數(ms): {r["repeat_median_values_ms"]}')
            print(f'  {"":<18} 平均±重複間std: {r["repeat_mean_ms"]:.1f}'
                  f'±{r["repeat_std_ms"]:.1f} ms '
                  f'(min {r["repeat_min_ms"]:.1f}, max {r["repeat_max_ms"]:.1f})')

    if len(results) >= 2:
        f, w = results[0], results[1]
        if args.repeats > 1:
            diff = f['repeat_mean_ms'] - w['repeat_mean_ms']
            combined_std = (f['repeat_std_ms'] ** 2 + w['repeat_std_ms'] ** 2) ** 0.5
            ratio = (abs(diff) / combined_std) if combined_std > 0 else float('inf')
            print(f'\nFULL − W2_semB（重複間平均差）：{diff:+.1f} ms')
            print(f'合併重複間標準差 sqrt(std_FULL^2+std_W2^2)：{combined_std:.1f} ms')
            print(f'|差距| / 合併標準差：{ratio:.2f}')
            if abs(diff) >= 3 * combined_std:
                verdict = '差距達合併重複間標準差 3 倍以上，可視為與量測雜訊可區分的效應。'
            elif abs(diff) >= 2 * combined_std:
                verdict = '差距達合併重複間標準差 2–3 倍，屬邊界情況，證據力有限。'
            else:
                verdict = '差距未達合併重複間標準差 2 倍，對齊網路的延遲成本無法與量測雜訊區分。'
            print(f'判定：{verdict}')
        else:
            d = f['median_ms'] - w['median_ms']
            print(f'\n參考影像路徑的延遲成本：{d:+.1f} ms '
                  f'({d / w["median_ms"] * 100:+.1f}%)')

    pre_align_row = None
    if args.bench_pre_align:
        full_label, full_run = RUNS[0]
        ckpt = os.path.join(ABL, full_run, 'weather_sam_best_latest.pth')
        if os.path.exists(ckpt):
            print(f'\n── pre_align 單獨計時：載入 {full_label} ({full_run})…', flush=True)
            model, cfg = load_pair_sam_from_ablation(ckpt, device=dev)
            batches = [make_batched_input(b, dev) for b in raw]
            lat, peak = bench_pre_align(model, batches, args.warmup, args.iters, dev)
            med = statistics.median(lat)
            pre_align_row = dict(
                label=f'{full_label} — pre_align only', run=full_run,
                median_ms=round(med, 2),
                mean_ms=round(statistics.mean(lat), 2),
                std_ms=round(statistics.pstdev(lat), 2),
                p90_ms=round(sorted(lat)[int(len(lat) * 0.9)], 2),
                peak_vram_mb=round(peak, 1),
            )
            print(f'   pre_align 延遲中位數 {pre_align_row["median_ms"]:.1f} ms  '
                  f'(p90 {pre_align_row["p90_ms"]:.1f}, '
                  f'平均 {pre_align_row["mean_ms"]:.1f}±{pre_align_row["std_ms"]:.1f})')
            del model, batches
            torch.cuda.empty_cache()
        else:
            print(f'[跳過 pre_align 計時] 找不到 {ckpt}')

    out = args.out or os.path.join(os.path.dirname(__file__), '..', '..',
                                   'figures_defense', 'inference_speed.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = dict(
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        warmup=args.warmup, iters=args.iters, repeats=args.repeats, input_size=1024,
        results=results,
    )
    if args.repeats > 1:
        payload['raw_repeats'] = {label: by_label[label] for label, run in RUNS
                                   if by_label[label]}
    if pre_align_row is not None:
        payload['pre_align_bench'] = pre_align_row
    with open(out, 'w') as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    print(f'\n已輸出：{out}')


if __name__ == '__main__':
    main()
