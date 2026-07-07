#!/usr/bin/env bash
# =============================================================================
# M-series Phase 1 — 關卡(FULL + B0,約 1 天)
#   FULL:e2e 權重 + 凍 encoder + decoder-ft + DeformAdapter(ref) + cond
#   B0  :同上但「無 adapter、無 cond」= 純解碼端域適應基線(累積表起點)
#
# 關卡判定(見規劃 §5):FULL 應顯著優於 B0,且 FULL 落在合理量級
# (參照 legacy FULL 67.26%);不過關先查管線,勿進 Phase 2。
# 假設已 conda activate sam_env。背景執行:
#   nohup bash scripts/ablation_m2f_phase1_gate.sh > outputs_ablation_m2f/phase1.log 2>&1 &
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ablation_m2f_common.sh
mkdir -p "$OUT_ROOT"

# FULL = B3 = T2(複用,消融表三處引用同一 run)
# 註:FULL 與 scripts/run_m2f_decoder_ft.sh 配置同構(seed 42);若該腳本已
#    跑完,可直接複用:cp -r outputs_m2f_decoder_ft outputs_ablation_m2f/FULL_seed42
run_one "FULL_seed42"

# B0:累積表起點(--no-use_vgg_adapter --no-cond 覆蓋 BASE 的預設開啟)
run_one "B0_seed42" --no-use_vgg_adapter --no-cond

echo "===== Phase 1 完成 ====="
print_summary
