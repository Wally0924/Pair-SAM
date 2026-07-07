#!/usr/bin/env bash
# =============================================================================
# M-series Phase 4 — 穩健性與配方(可選,約 2 天)
#   S1a/S1b:FULL @ seed 1234 / 2026(FULL 報 mean±std,協定沿用舊 campaign)
#   S2:FULL − label smoothing(--no-m2f_label_smooth)
#   S4:adapter_lr_scale 3.0 → 1.0(LR 敏感度)
#   註:原 S3(RCS)經使用者決定不做,已移除(舊 campaign 已證 MFB-only 較佳,
#      且 RCS 屬取樣配方而非新架構模組,對 M-series 主張無直接貢獻)。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ablation_m2f_common.sh
mkdir -p "$OUT_ROOT"

run_one "FULL_seed1234" --seed 1234
run_one "FULL_seed2026" --seed 2026
run_one "S2_nols_seed42" --no-m2f_label_smooth
run_one "S4_adplr1_seed42" --adapter_lr_scale 1.0

echo "===== Phase 4 完成 ====="
print_summary
