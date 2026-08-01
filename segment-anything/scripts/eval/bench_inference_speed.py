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

注意：本腳本量測分割模型本身；UAWarpC 對齊網路的成本以 FULL − W2_semB 的延遲差
估計（W2_semB 走 _adapter_reference_free 分支，不執行 pre_align，因此兩者延遲差
即為對齊網路加參考路徑的成本）。

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--n-batch', type=int, default=5, help='輪替使用的樣本數')
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

    results = []
    for label, run in RUNS:
        ckpt = os.path.join(ABL, run, 'weather_sam_best_latest.pth')
        if not os.path.exists(ckpt):
            print(f'[跳過] {label}: 找不到 {ckpt}')
            continue
        print(f'── {label} ({run}) 載入中…', flush=True)
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
        results.append(row)
        print(f'   延遲中位數 {row["median_ms"]:.1f} ms  '
              f'(p90 {row["p90_ms"]:.1f}, 平均 {row["mean_ms"]:.1f}±{row["std_ms"]:.1f})  '
              f'{row["fps"]:.2f} FPS')
        print(f'   峰值 VRAM {row["peak_vram_mb"]:.0f} MB   '
              f'參數 {row["params_total_m"]:.1f} M\n')

        del model, batches
        torch.cuda.empty_cache()

    print('=' * 78)
    print(f'{"設定":<18}{"延遲(ms)":>11}{"FPS":>8}{"VRAM(MB)":>11}{"參數(M)":>10}')
    print('-' * 78)
    for r in results:
        print(f'{r["label"]:<18}{r["median_ms"]:>11.1f}{r["fps"]:>8.2f}'
              f'{r["peak_vram_mb"]:>11.0f}{r["params_total_m"]:>10.1f}')
    print('=' * 78)

    if len(results) >= 2:
        f, w = results[0], results[1]
        d = f['median_ms'] - w['median_ms']
        print(f'\n參考影像路徑的延遲成本：{d:+.1f} ms '
              f'({d / w["median_ms"] * 100:+.1f}%)')

    out = args.out or os.path.join(os.path.dirname(__file__), '..', '..',
                                   'figures_defense', 'inference_speed.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fp:
        json.dump(dict(
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            warmup=args.warmup, iters=args.iters, input_size=1024,
            results=results,
        ), fp, indent=2, ensure_ascii=False)
    print(f'\n已輸出：{out}')


if __name__ == '__main__':
    main()
