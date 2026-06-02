#!/usr/bin/env bash
# 消融實驗：10 unique config / 16 訓練 run。R1/FULL/A2 各 3 seeds。
# C2 = R6（複用，不另訓）。每個 run 互相獨立，可平行。
#
# ⚠️ 本腳本為數天 GPU 算力的主體，請手動執行（可分批 / 背景跑）。
# 用法：bash run_ablation.sh   （或挑單行指令逐一執行）
set -euo pipefail
cd "$(dirname "$0")"

SEEDS_KEY=(42 1234 2026)        # R1 / FULL / A2 用 3 seeds
OUT=outputs_ablation
COMMON="--epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5"

run () { python train.py $COMMON "$@"; }

# ── 累積表 R1–R6（單 seed=42；R1 另跑 3 seeds）──
# R1 baseline：無 adapter / per-class / 純CE / 無LRH / 無MFB
for s in "${SEEDS_KEY[@]}"; do
  run --seed "$s" --no-use_vgg_adapter --decoder per_class --no-lrh --no-mfb \
      --lovasz_weight 0 --dice_weight 0 --output_dir "$OUT/R1_seed$s"
done
# R2 +Ref 後置注入（adapter 預設啟用）
run --seed 42 --inject post --decoder per_class --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir "$OUT/R2_seed42"
# R3 前置注入
run --seed 42 --inject pre --decoder per_class --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir "$OUT/R3_seed42"
# R4 統一查詢
run --seed 42 --inject pre --decoder unified --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir "$OUT/R4_seed42"
# R5 +LRH
run --seed 42 --inject pre --decoder unified --lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir "$OUT/R5_seed42"
# R6 +Lovász/Dice（= loss 表的 C2「取消 MFB」，複用）
run --seed 42 --inject pre --decoder unified --lrh --no-mfb \
    --lovasz_weight 1 --dice_weight 1 --output_dir "$OUT/R6_seed42"

# ── FULL（3 seeds）= R6 + MFB ──
for s in "${SEEDS_KEY[@]}"; do
  run --seed "$s" --inject pre --decoder unified --lrh --mfb \
      --lovasz_weight 1 --dice_weight 1 --output_dir "$OUT/FULL_seed$s"
done

# ── leave-one-out 變體 ──
# A1 後置注入（= FULL 但 inject post），單 seed
run --seed 42 --inject post --decoder unified --lrh --mfb \
    --lovasz_weight 1 --dice_weight 1 --output_dir "$OUT/A1_seed42"
# A2 移除 reference（= FULL 但 --no-ref），3 seeds
for s in "${SEEDS_KEY[@]}"; do
  run --seed "$s" --inject pre --decoder unified --lrh --mfb --no-ref \
      --lovasz_weight 1 --dice_weight 1 --output_dir "$OUT/A2_seed$s"
done
# C1 純 CE（= FULL 但 loss=CE only），單 seed
run --seed 42 --inject pre --decoder unified --lrh --mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir "$OUT/C1_seed42"

# ── 逐 run 評估 ──
for d in "$OUT"/*/; do
  ckpt="$d/weather_sam_best_latest.pth"
  if [[ -f "$ckpt" ]]; then
    python scripts/eval/eval_e1_acdc_val_full.py \
      --ckpt "$ckpt" --out "$d/e1_results.json" || echo "⚠️ eval failed: $d"
  else
    echo "⚠️ no checkpoint in $d (run may not have completed)"
  fi
done

# ── 彙整 3 張表 ──
python scripts/aggregate_ablation.py \
  --runs_root "$OUT" --out "$OUT/ablation_tables.tex"

echo "✅ all runs + eval + tables done → $OUT/ablation_tables.tex"
