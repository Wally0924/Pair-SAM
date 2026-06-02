#!/usr/bin/env bash
# 消融實驗：12 unique config / 14 訓練 run。RCS 為累積最後一步（R8=FULL）。
# 僅 R8(FULL) 跑 3 seeds；其餘各 1。R1–R7 為 --no-rcs；R8 與 A1/A2/C1/C2 為 --rcs。
# C2(取消MFB) 為獨立 run（mfb off, rcs on），不再複用 R6。
#
# ⚠️ 數天 GPU 算力主體，請手動執行（可分批 / 背景跑）。
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_CSV=/home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_train.csv
CP_JSON=/home/rvl1421/SAM_research-1/Datasets/class_presence.json
OUT=outputs_ablation
SEEDS_FULL=(42 1234 2026)        # 僅 FULL(R8) 用 3 seeds
COMMON="--epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5"

run () { python train.py $COMMON "$@"; }

# ── Step 0：precompute 每影像類別表（RCS 必需；冪等）──
python scripts/precompute_class_presence.py --csv "$TRAIN_CSV" --out "$CP_JSON"

# ── 累積表 R1–R7（單 seed=42，全部 --no-rcs）──
# R1 baseline：無 adapter / per-class / 純CE / 無LRH / 無MFB / 無RCS
run --seed 42 --no-use_vgg_adapter --decoder per_class --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --no-rcs --output_dir "$OUT/R1_seed42"
# R2 +Ref（後置注入）
run --seed 42 --inject post --decoder per_class --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --no-rcs --output_dir "$OUT/R2_seed42"
# R3 前置注入
run --seed 42 --inject pre --decoder per_class --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --no-rcs --output_dir "$OUT/R3_seed42"
# R4 統一查詢
run --seed 42 --inject pre --decoder unified --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --no-rcs --output_dir "$OUT/R4_seed42"
# R5 +LRH
run --seed 42 --inject pre --decoder unified --lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --no-rcs --output_dir "$OUT/R5_seed42"
# R6 +Lovász/Dice
run --seed 42 --inject pre --decoder unified --lrh --no-mfb \
    --lovasz_weight 1 --dice_weight 1 --no-rcs --output_dir "$OUT/R6_seed42"
# R7 +MFB（= 舊 FULL；= 新 FULL 去 RCS，即 RCS 控制組）
run --seed 42 --inject pre --decoder unified --lrh --mfb \
    --lovasz_weight 1 --dice_weight 1 --no-rcs --output_dir "$OUT/R7_seed42"

# ── R8 = FULL（+RCS）3 seeds ──
for s in "${SEEDS_FULL[@]}"; do
  run --seed "$s" --inject pre --decoder unified --lrh --mfb \
      --lovasz_weight 1 --dice_weight 1 --rcs --output_dir "$OUT/R8_seed$s"
done

# ── leave-one-out（皆相對 R8，rcs on，單 seed）──
# A1 後置注入（= R8 但 inject post）
run --seed 42 --inject post --decoder unified --lrh --mfb \
    --lovasz_weight 1 --dice_weight 1 --rcs --output_dir "$OUT/A1_seed42"
# A2 移除 reference（= R8 但 --no-ref）
run --seed 42 --inject pre --decoder unified --lrh --mfb --no-ref \
    --lovasz_weight 1 --dice_weight 1 --rcs --output_dir "$OUT/A2_seed42"
# C1 純 CE（= R8 但 loss=CE only）
run --seed 42 --inject pre --decoder unified --lrh --mfb \
    --lovasz_weight 0 --dice_weight 0 --rcs --output_dir "$OUT/C1_seed42"
# C2 取消 MFB（= R8 但 --no-mfb；mfb off, rcs on，獨立 run）
run --seed 42 --inject pre --decoder unified --lrh --no-mfb \
    --lovasz_weight 1 --dice_weight 1 --rcs --output_dir "$OUT/C2_seed42"

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
python scripts/aggregate_ablation.py --runs_root "$OUT" --out "$OUT/ablation_tables.tex"

echo "✅ 14 runs + eval + tables done → $OUT/ablation_tables.tex"
