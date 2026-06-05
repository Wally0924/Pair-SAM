#!/usr/bin/env bash
# 消融實驗：10 unique config / 12 訓練 run。FULL = R7（MFB-only，無 RCS）。
# 對照實驗（docs/experiments/2026-06-06-mfb-vs-rcs-comparison.md）顯示 MFB-only 最佳，RCS 已移除。
# 僅 R7(FULL) 跑 3 seeds；其餘各 1。C2(取消MFB) = R6（同 config，免費複用，不另訓）。
#
# ⚠️ 數天 GPU 算力主體，請手動執行（可分批 / 背景跑）。冪等：已有 checkpoint 的 run 自動略過。
set -uo pipefail
cd "$(dirname "$0")"

OUT=outputs_ablation
SEEDS_FULL=(42 1234 2026)        # 僅 FULL(R7) 用 3 seeds
COMMON="--epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5"

mkdir -p "$OUT"
# 冪等訓練：已有 weather_sam_best_latest.pth 就跳過
run () {
  dir="$1"; shift
  if [[ -f "$OUT/$dir/weather_sam_best_latest.pth" ]]; then
    echo "⏩ skip $dir（已完成）"; return 0
  fi
  python train.py $COMMON "$@" --output_dir "$OUT/$dir" || echo "⚠️ train failed: $dir（繼續）"
}

# ── 累積表 R1–R6（單 seed=42；全部不用 RCS，rcs 預設已為 off）──
# R1 baseline：無 adapter / per-class / 純CE / 無LRH / 無MFB
run R1_seed42 --seed 42 --no-use_vgg_adapter --decoder per_class --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R2 +Ref（後置注入）
run R2_seed42 --seed 42 --inject post --decoder per_class --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R3 前置注入
run R3_seed42 --seed 42 --inject pre  --decoder per_class --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R4 統一查詢
run R4_seed42 --seed 42 --inject pre  --decoder unified   --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R5 +LRH
run R5_seed42 --seed 42 --inject pre  --decoder unified   --lrh    --no-mfb --lovasz_weight 0 --dice_weight 0
# R6 +Lovász/Dice（= loss 表的 C2「取消 MFB」，複用，不另訓）
run R6_seed42 --seed 42 --inject pre  --decoder unified   --lrh    --no-mfb --lovasz_weight 1 --dice_weight 1

# ── R7 = FULL（+MFB）3 seeds ──
for s in "${SEEDS_FULL[@]}"; do
  run R7_seed$s --seed "$s" --inject pre --decoder unified --lrh --mfb --lovasz_weight 1 --dice_weight 1
done

# ── leave-one-out（皆相對 R7，單 seed）──
# A1 後置注入（= R7 但 inject post）
run A1_seed42 --seed 42 --inject post --decoder unified --lrh --mfb    --lovasz_weight 1 --dice_weight 1
# A2 移除 reference（= R7 但 --no-ref）
run A2_seed42 --seed 42 --inject pre  --decoder unified --lrh --mfb --no-ref --lovasz_weight 1 --dice_weight 1
# C1 純 CE（= R7 但 loss=CE only）
run C1_seed42 --seed 42 --inject pre  --decoder unified --lrh --mfb    --lovasz_weight 0 --dice_weight 0
# C2 取消 MFB = R6（同 config，免費複用；不另訓）

# ── 逐 run 評估 ──
for d in "$OUT"/*/; do
  ckpt="$d/weather_sam_best_latest.pth"
  if [[ -f "$ckpt" ]]; then
    [[ -f "$d/e1_results.json" ]] || python scripts/eval/eval_e1_acdc_val_full.py \
      --ckpt "$ckpt" --out "$d/e1_results.json" || echo "⚠️ eval failed: $d"
  else
    echo "⚠️ no checkpoint in $d"
  fi
done

# ── 彙整 3 張表 ──
python scripts/aggregate_ablation.py --runs_root "$OUT" --out "$OUT/ablation_tables.tex"

echo "✅ 12 runs + eval + tables done → $OUT/ablation_tables.tex"
