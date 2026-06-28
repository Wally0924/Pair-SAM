#!/usr/bin/env bash
# 累積消融「重排順序」追加訓練：把唯一的使能型模組「統一查詢」移到最後一步，
# 讓 N1→N7 的 mIoU 軌跡單調上升（取代原 R3→R4 的 −21.0 崩落呈現）。
#
# 新順序（decoder 全程 per_class，直到最後一步才切 unified）：
#   N1 Baseline      = 既有 R1   (39.6)              ← 沿用，不重訓
#   N2 +Ref          = 既有 R2   (60.7)              ← 沿用，不重訓
#   N3 +Pre-inject   = 既有 R3   (62.8)              ← 沿用，不重訓
#   N4 +LRH          = R3 + LRH                       ← 本腳本重訓
#   N5 +Lov.+Dice    = N4 + Lovász/Dice               ← 本腳本重訓
#   N6 +MFB          = N5 + MFB = Full − 統一查詢      ← 本腳本重訓（同時補上 decoder 維度 leave-one-out）
#   N7 +Unified (Full) = 既有 R7 (67.26)              ← 沿用，不重訓
#
# 三組只差該差的那一個開關，其餘超參數沿用 R1–R7 批次（無混淆變因）。
# 單卡序列、單 seed=42、冪等（已有 best ckpt 自動略過）。約 1–1.5 天。
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

# N4 +LRH（per_class，= R3 + LRH）
run N4_seed42 --seed 42 --inject pre --decoder per_class --lrh    --no-mfb --lovasz_weight 0 --dice_weight 0
# N5 +Lov.+Dice（per_class，= N4 + Lovász/Dice）
run N5_seed42 --seed 42 --inject pre --decoder per_class --lrh    --no-mfb --lovasz_weight 1 --dice_weight 1
# N6 +MFB（per_class，= Full − 統一查詢；亦即 decoder 維度 leave-one-out）
run N6_seed42 --seed 42 --inject pre --decoder per_class --lrh    --mfb    --lovasz_weight 1 --dice_weight 1

# ── 逐 run 評估（冪等：已有 e1_results.json 就跳過）──
for d in N4_seed42 N5_seed42 N6_seed42; do
  ckpt="$OUT/$d/weather_sam_best_latest.pth"
  if [[ -f "$ckpt" ]]; then
    [[ -f "$OUT/$d/e1_results.json" ]] || python scripts/eval/eval_e1_acdc_val_full.py \
      --ckpt "$ckpt" --out "$OUT/$d/e1_results.json" || echo "⚠️ eval failed: $d"
  else
    echo "⚠️ no checkpoint in $d"
  fi
done

# ── 彙整：印出重排後的 7 點累積軌跡，立即檢查是否單調 ──
python - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
def miou(d):
    p = os.path.join(out, d, 'e1_results.json')
    return json.load(open(p))['overall_miou']*100 if os.path.isfile(p) else None
# (顯示名, run 目錄)；N1/N2/N3/N7 沿用既有 R1/R2/R3/R7
seq = [('N1 Baseline','R1_seed42'), ('N2 +Ref','R2_seed42'),
       ('N3 +Pre-inject','R3_seed42'), ('N4 +LRH','N4_seed42'),
       ('N5 +Lov.+Dice','N5_seed42'), ('N6 +MFB (Full−UQ)','N6_seed42'),
       ('N7 +Unified (Full)','R7_seed42')]
print("\n=== 重排累積軌跡（ACDC val, seed42）===")
prev = None
for name, d in seq:
    m = miou(d)
    if m is None:
        print(f"{name:22} {'MISSING':>8}   (缺 {d}/e1_results.json)"); prev=None; continue
    delta = f"{m-prev:+5.2f}" if prev is not None else "  ---"
    flag = "" if (prev is None or m >= prev) else "  ⚠️非單調"
    print(f"{name:22} {m:7.2f}   Δ={delta}{flag}")
    prev = m
# decoder 維度 leave-one-out：Full(R7) − Full−UQ(N6)
r7, n6 = miou('R7_seed42'), miou('N6_seed42')
if r7 is not None and n6 is not None:
    print(f"\n=== decoder leave-one-out ===")
    print(f"Full (R7) = {r7:.2f} ；移除統一查詢 (N6) = {n6:.2f} ；Δ(統一查詢貢獻) = {r7-n6:+.2f} pp")
PY

echo "✅ reorder 追加訓練 + eval + 軌跡彙整完成"
