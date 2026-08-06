#!/usr/bin/env bash
# =============================================================================
# M-series Phase 6 — 同影像先驗基線（P1，約 8 小時）
#
# 目的：產出與完整模型只差單一變因的受控基線。P1 將 DeformAdapter 的先驗來源
#   由「UAWarpC 對齊後的跨視角晴天參考」換成「當前影像」（ViT-Adapter 式 SPM），
#   其餘主幹、解碼器、閘控零初始化、注入位置、抽取器、資料、排程、seed 全部相同。
#
# 回答兩個問題：
#   1. 先驗該取自跨視角參考還是當前影像？（對照 FULL_seed42 = 76.02）
#   2. W4_seed42 的 79.80 是否源自其固定 0.05 閘控初始化？P1 用零初始化閘控，
#      若 P1 亦顯著高於 FULL，則排除閘控假說。
#
# 設計文件：docs/superpowers/specs/2026-08-06-selfprior-baseline-design.md
#
# 假設已 conda activate sam_env。背景執行：
#   nohup bash scripts/ablation_m2f_phase6_selfprior.sh > outputs_ablation_m2f/phase6.log 2>&1 &
#
# 注意：本腳本只跑訓練與 ACDC val 評估。ACDC test 提交須另行執行
#   scripts/eval/dump_acdc_test_preds.py，且會消耗不可逆的官方 server 配額。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ablation_m2f_common.sh
mkdir -p "$OUT_ROOT"

run_one "P1_selfprior_seed42" --prior_source self

echo "===== Phase 6 完成 ====="
print_summary
