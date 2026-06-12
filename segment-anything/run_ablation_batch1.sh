#!/usr/bin/env bash
# 消融實驗 Batch 1/2：端點與控制組優先（runbook Phase 3a 原則）。
# A2(移除ref，中心論點) → R1(baseline錨點) → R6(=C2複用) → C1(純CE)。
# 跑完本批：loss 表（FULL/C1/C2）即完整、A2 落差可先驗證。
# R7 三 seeds 已完成（冪等自動略過）。約 1–1.5 天。單卡序列執行。
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

# A2 移除 reference（= R7 但 --no-ref，中心論點控制組）
run A2_seed42 --seed 42 --inject pre  --decoder unified --lrh    --mfb --no-ref --lovasz_weight 1 --dice_weight 1
# R1 baseline：無 adapter / per-class / 純CE / 無LRH / 無MFB
run R1_seed42 --seed 42 --no-use_vgg_adapter --decoder per_class --no-lrh --no-mfb --lovasz_weight 0 --dice_weight 0
# R6 +Lovász/Dice（= loss 表的 C2「取消 MFB」，複用，不另訓）
run R6_seed42 --seed 42 --inject pre  --decoder unified   --lrh    --no-mfb --lovasz_weight 1 --dice_weight 1
# C1 純 CE（= R7 但 loss=CE only）
run C1_seed42 --seed 42 --inject pre  --decoder unified   --lrh    --mfb    --lovasz_weight 0 --dice_weight 0

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

# 先行彙整（缺的列會印 missing 註解，供提早檢視 loss 表與 A2 落差）
python scripts/aggregate_ablation.py --runs_root "$OUT" --out "$OUT/ablation_tables.tex" || true

echo "✅ Batch 1 done（A2/R1/R6/C1 + eval）→ 接著跑 run_ablation_batch2.sh"
