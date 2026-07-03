#!/usr/bin/env bash
# R7 歸因消融：拆分「架構效益 vs 初始化效益」，兩臂共用同一配方與 seed。
#   R7sam  = 新 DeformAdapter 架構 + SAM 原始權重（sam_vit_h_4b8939.pth）
#   R7city = 新 DeformAdapter 架構 + Cityscapes fine-tune encoder（*_cityscapes_merged.pth）
#   架構效益   = R7sam  − 舊R7(0.6725)
#   初始化效益 = R7city − R7sam
# ⚠️ 兩臂皆用當前 code（雙向端到端梯度 + dice 無 MFB 加權）。舊 R7_seed42 是舊配方，
#    只能當參考；嚴謹歸因請以本腳本重跑的 R7sam/R7city 為準。
# 開訓需看到「Missing keys ...: 966」，與舊 R7 相同才代表載入正確。
# 用法：bash run_r7_attribution.sh [city|sam|both]   （預設 both；sam 先跑）
set -uo pipefail
cd "$(dirname "$0")"

PY="conda run --no-capture-output -n sam_env python"
# 減少 CUDA allocator 碎片化（OOM 當下有 ~1GB reserved-but-unallocated）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run_arm <name> <ckpt> <outdir>
run_arm() {
    local name="$1" ckpt="$2" out="$3"
    if [ ! -f "$ckpt" ]; then echo "❌ 找不到權重 $ckpt"; exit 1; fi
    mkdir -p "$out"
    echo "=================== $name (train) ==================="
    if $PY train.py \
        --epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
        --inject pre --decoder unified --lrh --mfb --no-rcs \
        --lovasz_weight 1 --dice_weight 1 --seed 42 \
        --checkpoint "$ckpt" \
        --output_dir "$out"; then
        echo "=================== $name (eval E1) ==================="
        $PY scripts/eval/eval_e1_acdc_val_full.py \
            --ckpt "$out/weather_sam_best_latest.pth" \
            --out  "$out/e1_results.json" || echo "⚠️ eval failed: $name"
    else
        echo "⚠️ train failed: $name"; exit 1
    fi
}

ARM="${1:-both}"
if [ "$ARM" != "city" ] && [ "$ARM" != "sam" ] && [ "$ARM" != "both" ]; then
    echo "❌ 用法：bash run_r7_attribution.sh [city|sam|both]"; exit 1
fi

if [ "$ARM" = "sam" ] || [ "$ARM" = "both" ]; then
    run_arm R7sam_seed42  checkpoints/sam_vit_h_4b8939.pth \
            outputs_ablation/R7sam_seed42
fi
if [ "$ARM" = "city" ] || [ "$ARM" = "both" ]; then
    run_arm R7city_seed42 checkpoints/cityscapes_pretrain/sam_vit_h_cityscapes_merged.pth \
            outputs_ablation/R7city_seed42
fi

echo
echo "✅ done。歸因拆分："
$PY - <<'PYEOF'
import json, os
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
sam  = load('outputs_ablation/R7sam_seed42/e1_results.json')
city = load('outputs_ablation/R7city_seed42/e1_results.json')
old  = load('outputs_ablation/R7_seed42/e1_results.json')
if sam:
    print(f"  R7sam  overall: {sam['overall_miou']*100:.2f}%")
    for k, v in sam['per_condition_miou'].items():
        print(f"    {k}: {v*100:.2f}%")
if city:
    print(f"  R7city overall: {city['overall_miou']*100:.2f}%")
    for k, v in city['per_condition_miou'].items():
        print(f"    {k}: {v*100:.2f}%")
if sam and old:
    print(f"  架構效益   = {sam['overall_miou']*100:.2f} − {old['overall_miou']*100:.2f}"
          f" = {(sam['overall_miou']-old['overall_miou'])*100:+.2f} pp（vs 舊 R7）")
if city and sam:
    print(f"  初始化效益 = {city['overall_miou']*100:.2f} − {sam['overall_miou']*100:.2f}"
          f" = {(city['overall_miou']-sam['overall_miou'])*100:+.2f} pp（vs R7sam）")
PYEOF
