#!/usr/bin/env bash
# 消融實驗追加：P1（condition embedding 消融，spec §2.7 / §6 第 6 步）。
# P1 = R7/FULL config 但 --no-cond（condition_id 固定共享索引，ConditionEncoder 退化為常數）。
# 單 seed=42，於 12 個主 run 完成後追加。結果以正文敘述（FULL vs P1），不進三張主表。
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

# P1（= R7 但 --no-cond）
run P1_seed42 --seed 42 --inject pre --decoder unified --lrh --mfb --no-cond --lovasz_weight 1 --dice_weight 1

# ── 評估（冪等）──
ckpt="$OUT/P1_seed42/weather_sam_best_latest.pth"
if [[ -f "$ckpt" ]]; then
  [[ -f "$OUT/P1_seed42/e1_results.json" ]] || python scripts/eval/eval_e1_acdc_val_full.py \
    --ckpt "$ckpt" --out "$OUT/P1_seed42/e1_results.json" || echo "⚠️ eval failed: P1_seed42"
else
  echo "⚠️ no checkpoint in P1_seed42"
fi

# ── 正文用對照：FULL(=R7_seed42) vs P1，overall + per-condition mIoU ──
python - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
def load(d):
    p = os.path.join(out, d, 'e1_results.json')
    return json.load(open(p)) if os.path.isfile(p) else None
full, p1 = load('R7_seed42'), load('P1_seed42')
if not (full and p1):
    print("⚠️ 缺 R7_seed42 或 P1_seed42 的 e1_results.json，跳過對照"); raise SystemExit
conds = ['fog', 'rain', 'snow', 'night']
print("\n=== FULL vs P1（condition embedding 消融）===")
print(f"{'':6} {'overall':>8} " + " ".join(f"{c:>7}" for c in conds))
for name, d in [('FULL', full), ('P1', p1)]:
    pc = d['per_condition_miou']
    row = " ".join(f"{pc.get(c, float('nan'))*100:7.2f}" for c in conds)
    print(f"{name:6} {d['overall_miou']*100:8.2f} {row}")
dlt = (p1['overall_miou'] - full['overall_miou']) * 100
print(f"Δ(P1−FULL) overall = {dlt:+.2f} pp")
PY

echo "✅ P1 done（train + eval + FULL/P1 對照）"
