#!/usr/bin/env bash
# 新 FULL = R7 配置（MFB-only，無 RCS）。補跑 seed 1234 / 2026，與已有的 R7_seed42 湊 3 seeds。
# 跑完自動 eval，並印出 3-seed overall mIoU 的 mean±std。
# 用法：bash run_full_seeds.sh
set -uo pipefail
cd "$(dirname "$0")"

OUT=outputs_ablation
# 新 FULL（R7）配置：MFB on、RCS off、其餘 = FULL
COMMON="--epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 --inject pre --decoder unified --lrh --mfb --no-rcs --lovasz_weight 1 --dice_weight 1"

mkdir -p "$OUT"

run_one () {
  s="$1"
  echo "=================== R7 seed=$s ==================="
  if python train.py $COMMON --seed "$s" --output_dir "$OUT/R7_seed$s"; then
    python scripts/eval/eval_e1_acdc_val_full.py \
      --ckpt "$OUT/R7_seed$s/weather_sam_best_latest.pth" \
      --out  "$OUT/R7_seed$s/e1_results.json" || echo "⚠️ eval failed: R7_seed$s"
  else
    echo "⚠️ train failed: R7_seed$s（繼續下一個）"
  fi
}

run_one 1234
run_one 2026

echo
echo "✅ done。R7（新 FULL）3-seed overall mIoU："
python - <<'PY'
import json, os, statistics
vals=[]
for s in (42,1234,2026):
    p=f"outputs_ablation/R7_seed{s}/e1_results.json"
    if os.path.isfile(p):
        m=json.load(open(p))['overall_miou']*100
        vals.append(m); print(f"  seed {s}: {m:.2f}%")
    else:
        print(f"  seed {s}: (無 e1_results.json)")
if len(vals)>=2:
    mean=statistics.mean(vals); std=statistics.stdev(vals)
    print(f"  >>> mean±std = {mean:.2f} ± {std:.2f}  (n={len(vals)})")
elif vals:
    print(f"  >>> 僅 {len(vals)} seed，無法算 std")
PY
