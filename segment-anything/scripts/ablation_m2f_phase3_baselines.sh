#!/usr/bin/env bash
# =============================================================================
# M-series Phase 3 — 基線對照與訓練策略(W4/W5/T1/T3,約 2 天)
#   W4:adapter → SAM-Adapter 同影像基線(adapter_variant=sam_adapter,
#       17.42M 與 DeformAdapter 17.49M 參數量對齊;2026-07-07 已完成 A3 hook
#       API 遷移 + builder rpm 防護,1-epoch smoke 與 eval 重建驗證通過)
#   W5:FULL − CS 預訓練(SAM 原權重起訓;解碼端從零。與其他 run 同預算
#       30 epochs——正文須標註「同預算比較」,見規劃 §5 誠實性)
#   T1:PEFT(--freeze_decoder,只訓 adapter+cond,17.5M/2.1%)
#   T3:全解凍上界(32 blocks + encoder_lr_scale 0.1,對齊 e2e 配方)
#
# ⚠️ T3 為本系列 VRAM 最重配置(全解凍 + adapter),尚未 memcheck 實測;
#    若 OOM,先跑 scripts/deform_adapter_memcheck.py 檢查,或單獨降載重跑。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ablation_m2f_common.sh
mkdir -p "$OUT_ROOT"

run_one "W4_seed42" --adapter_variant sam_adapter
run_one "W5_seed42" --checkpoint "$SAM_CKPT"
run_one "T1_seed42" --freeze_decoder
run_one "T3_seed42" --unfreeze_encoder_blocks 32 --encoder_lr_scale 0.1

echo "===== Phase 3 完成 ====="
print_summary
