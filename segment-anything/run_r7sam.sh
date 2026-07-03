#!/usr/bin/env bash
# R7sam：新 DeformAdapter 架構 + SAM 原始權重（sam_vit_h_4b8939.pth）。
# 目的：與 R7city（CS fine-tune encoder）拆分「架構效益 vs 初始化效益」：
#   架構效益   = R7sam − 舊R7(0.6725)
#   初始化效益 = R7city − R7sam
# ⚠️ 配方版本：本 run 使用當前 code（dice 無 MFB 加權）。若要與 R7city 對照，
#    R7city 必須用同版 code 重跑，否則 loss 配方是額外變因。
# 用法：bash run_r7sam.sh
set -uo pipefail
cd "$(dirname "$0")"

OUT=outputs_ablation/R7sam_seed42
CKPT=checkpoints/sam_vit_h_4b8939.pth
PY="conda run --no-capture-output -n sam_env python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$CKPT" ]; then
    echo "❌ 找不到 SAM 權重 $CKPT"; exit 1
fi
mkdir -p "$OUT"

echo "=================== R7sam seed=42 (train) ==================="
if $PY train.py \
    --epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
    --inject pre --decoder unified --lrh --mfb --no-rcs \
    --lovasz_weight 1 --dice_weight 1 --seed 42 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT"; then
    echo "=================== R7sam seed=42 (eval E1) ==================="
    $PY scripts/eval/eval_e1_acdc_val_full.py \
        --ckpt "$OUT/weather_sam_best_latest.pth" \
        --out  "$OUT/e1_results.json" || echo "⚠️ eval failed: R7sam_seed42"
else
    echo "⚠️ train failed: R7sam_seed42"; exit 1
fi

echo
echo "✅ done。歸因拆分："
$PY - <<'PYEOF'
import json
r = json.load(open('outputs_ablation/R7sam_seed42/e1_results.json'))
print(f"  R7sam  overall: {r['overall_miou']*100:.2f}%")
for k, v in r['per_condition_miou'].items():
    print(f"    {k}: {v*100:.2f}%")
try:
    old = json.load(open('outputs_ablation/R7_seed42/e1_results.json'))
    print(f"  架構效益   = {r['overall_miou']*100:.2f} − {old['overall_miou']*100:.2f}"
          f" = {(r['overall_miou']-old['overall_miou'])*100:+.2f} pp（vs 舊 R7）")
except FileNotFoundError:
    pass
try:
    city = json.load(open('outputs_ablation/R7city_seed42/e1_results.json'))
    print(f"  初始化效益 = {city['overall_miou']*100:.2f} − {r['overall_miou']*100:.2f}"
          f" = {(city['overall_miou']-r['overall_miou'])*100:+.2f} pp（vs R7city）")
except FileNotFoundError:
    print("  （R7city 尚無 e1_results.json，跑完後再對照初始化效益）")
PYEOF
