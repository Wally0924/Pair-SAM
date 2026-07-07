#!/usr/bin/env bash
# =============================================================================
# M-series Phase 2 — Adapter 解剖(B1/B2/W1/W2,約 2 天)
#   B1:adapter 容量、零參考資訊(--no-ref)、無 cond   → 容量 vs 資訊
#   B2:+ 參考資訊、無 cond(= W3 複用)               → C1 核心證據
#   W1:FULL − adapter(cond 保留)                     → leave-one-out Δ(adapter)
#   W2:FULL − ref(cond 保留)                         → leave-one-out Δ(參考資訊)
#
# 方向確認:B2 > B1(參考資訊 > 純容量);若不成立,據實寫,不調數據。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ablation_m2f_common.sh
mkdir -p "$OUT_ROOT"

run_one "B1_seed42" --no-ref --no-cond
run_one "B2_seed42" --no-cond
run_one "W1_seed42" --no-use_vgg_adapter
run_one "W2_seed42" --no-ref

echo "===== Phase 2 完成 ====="
print_summary
