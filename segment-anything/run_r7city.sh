#!/usr/bin/env bash
# R7city：新 FULL（R7）配置 + Cityscapes Stage-1 encoder（decoder = SAM pretrain）。
# 權重 = sam_vit_h_cityscapes_merged.pth（SAM 打底、image_encoder.* 覆蓋為 Cityscapes 版）。
# 與舊 R7 唯一變因 = encoder 初始化；輸出到 R7city_seed42，不動論文引用的 R7_seed42。
# 開訓需看到「Missing keys ...: 966」，與舊 R7 相同才代表載入正確。
# 用法：bash run_r7city.sh
set -uo pipefail
cd "$(dirname "$0")"

OUT=outputs_ablation/R7city_seed42
CKPT=checkpoints/cityscapes_pretrain/sam_vit_h_cityscapes_merged.pth
PY="conda run --no-capture-output -n sam_env python"
# 減少 CUDA allocator 碎片化（OOM 當下有 ~1GB reserved-but-unallocated）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$CKPT" ]; then
    echo "❌ 找不到合併權重 $CKPT"; exit 1
fi
mkdir -p "$OUT"

echo "=================== R7city seed=42 (train) ==================="
if $PY train.py \
    --epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
    --inject pre --decoder unified --lrh --mfb --no-rcs \
    --lovasz_weight 1 --dice_weight 1 --seed 42 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT"; then
    echo "=================== R7city seed=42 (eval E1) ==================="
    $PY scripts/eval/eval_e1_acdc_val_full.py \
        --ckpt "$OUT/weather_sam_best_latest.pth" \
        --out  "$OUT/e1_results.json" || echo "⚠️ eval failed: R7city_seed42"
else
    echo "⚠️ train failed: R7city_seed42"; exit 1
fi

echo
echo "✅ done。R7city overall mIoU："
$PY - <<'PYEOF'
import json
r = json.load(open('outputs_ablation/R7city_seed42/e1_results.json'))
print(f"  overall: {r['overall_miou']*100:.2f}%")
for k, v in r['per_condition_miou'].items():
    print(f"  {k}: {v*100:.2f}%")
old = json.load(open('outputs_ablation/R7_seed42/e1_results.json'))
print(f"  (舊 R7_seed42 overall: {old['overall_miou']*100:.2f}%)")
PYEOF
