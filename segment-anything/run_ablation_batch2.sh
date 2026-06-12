#!/usr/bin/env bash
# 消融實驗 Batch 2/2：累積表中間列 + A1。先跑完 run_ablation_batch1.sh。
# R2 → R3 → R4 → R5 → A1，結束後全量 eval + 彙整 3 張表。約 1.5–2 天。單卡序列執行。
set -uo pipefail
cd "$(dirname "$0")"

OUT=outputs_ablation
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

# R2 +Ref（後置注入）
run R2_seed42 --seed 42 --inject post --decoder per_class --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R3 前置注入
run R3_seed42 --seed 42 --inject pre  --decoder per_class --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R4 統一查詢
run R4_seed42 --seed 42 --inject pre  --decoder unified   --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R5 +LRH
run R5_seed42 --seed 42 --inject pre  --decoder unified   --lrh    --no-mfb --lovasz_weight 0 --dice_weight 0
# A1 後置注入（= R7 但 inject post）
run A1_seed42 --seed 42 --inject post --decoder unified   --lrh    --mfb    --lovasz_weight 1 --dice_weight 1

# ── 逐 run 評估（冪等：已有 e1_results.json 就跳過）──
for d in "$OUT"/*/; do
  ckpt="$d/weather_sam_best_latest.pth"
  if [[ -f "$ckpt" ]]; then
    [[ -f "$d/e1_results.json" ]] || python scripts/eval/eval_e1_acdc_val_full.py \
      --ckpt "$ckpt" --out "$d/e1_results.json" || echo "⚠️ eval failed: $d"
  else
    echo "⚠️ no checkpoint in $d"
  fi
done

# ── 彙整 3 張表（全 12 runs 到齊）──
python scripts/aggregate_ablation.py --runs_root "$OUT" --out "$OUT/ablation_tables.tex"

echo "✅ 12 runs + eval + tables done → $OUT/ablation_tables.tex"
