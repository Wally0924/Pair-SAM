# =============================================================================
# M-series 消融共用設定(被 ablation_m2f_phase*.sh source,不直接執行)
# 規劃文件:docs/experiments/2026-07-06-m2f-ablation-plan.md
#
# 協定:所有 run 共享 BASE_FLAGS(§4.0);消融軸以追加旗標覆蓋(argparse 取最後值)。
# 冪等:已有 best ckpt 的 run 跳過訓練;已有 e1_results.json 的 run 跳過 eval。
# =============================================================================

E2E_CKPT="/home/rvl1421/SAM_research-1/segment-anything/outputs_m2f_cs_e2e/weather_sam_best_latest.pth"
SAM_CKPT="/home/rvl1421/SAM_research-1/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
OUT_ROOT="outputs_ablation_m2f"

# §4.0 共同協定(全部寫死;死旗標 lovasz_weight 固定 0.0 僅為 config 落地誠實)
BASE_FLAGS=(
  --model_type vit_h
  --decoder m2f
  --checkpoint "$E2E_CKPT"
  --use_vgg_adapter
  --unfreeze_encoder_blocks 0
  --epochs 30
  --patience 10
  --batch_size 1
  --accumulate_steps 4
  --lr 5e-5
  --weight_decay 0.05
  --max_norm 1.0
  --warmup_epochs 5
  --warmup_gate_epochs 0
  --decoder_lr_scale 1.0
  --adapter_lr_scale 3.0
  --dice_weight 5.0
  --cls_weight 2.0
  --bce_weight 5.0
  --no_object_weight 0.1
  --num_points 12544
  --lovasz_weight 0.0
  --m2f_label_smooth
  --label_smoothing 0.05
  --seed 42
)

# run_one <RunID> [覆蓋旗標...]:訓練(若無 ckpt)+ eval(若無結果)
run_one() {
  local run_id="$1"; shift
  local dir="${OUT_ROOT}/${run_id}"
  mkdir -p "$dir"
  if [ -f "${dir}/weather_sam_best_latest.pth" ]; then
    echo "⏭️  [${run_id}] checkpoint 已存在,跳過訓練"
  else
    echo "🚀 [${run_id}] 訓練開始:$(date '+%F %T')"
    python -u train.py "${BASE_FLAGS[@]}" --output_dir "$dir" "$@"
  fi
  if [ ! -f "${dir}/e1_results.json" ]; then
    echo "📊 [${run_id}] ACDC val 評估"
    python scripts/eval/eval_e1_acdc_val_full.py \
      --ckpt "${dir}/weather_sam_best_latest.pth" \
      --out  "${dir}/e1_results.json"
  fi
}

# 印出 OUT_ROOT 下所有已完成 run 的 mIoU 摘要
print_summary() {
  python - <<'PY'
import json, glob, os
paths = sorted(glob.glob('outputs_ablation_m2f/*/e1_results.json'))
if not paths:
    print('(尚無完成的 run)')
for p in paths:
    d = json.load(open(p))
    run = os.path.basename(os.path.dirname(p))
    cond = '  '.join(f'{k}={v*100:.2f}' for k, v in d.get('per_condition_miou', {}).items())
    print(f"{run:22s} overall={d['overall_miou']*100:.2f}%  {cond}")
PY
}
