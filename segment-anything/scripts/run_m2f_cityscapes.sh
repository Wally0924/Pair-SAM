#!/usr/bin/env bash
# =============================================================================
# PairSAM M2F 訓練：PEFT 天氣適應階段
#   來源模型 = outputs_m2f_cs_e2e（Cityscapes 晴天端到端預訓練的 encoder + MSDeformAttn
#   pixel decoder + masked-attention decoder）。本階段載入該完整權重，凍結 encoder+decoder，
#   在 ACDC 惡劣天氣上「只訓練額外天氣模組」：DeformAdapter（vgg_injector）+ condition_encoder。
#
# 參數來源稽核（2026-07-04）：
#   [M2F 官方 Mask2Former] cls=2.0 / bce=5.0 / dice=5.0 / no_object=0.1 /
#       num_points=12544 / oversample=3.0 / importance=0.75（後三者為 m2f_loss 內建預設）
#   [對齊官方] weight_decay=0.05（Mask2Former SOLVER.WEIGHT_DECAY）
#   [保留現值] max_norm=1.0（官方 0.01 為 ResNet/Swin 全訓練調校，不套用於本凍結設定）
#   [設計依據] lr=5e-5（PEFT 適應 base LR，adapter 另乘 adapter_lr_scale 預設 3×）/
#       Warmup+Cosine / eff.batch=4（4090）/ lovasz=0.0（m2f 不走 ContextLoss）
#   [凍結策略] --freeze_decoder + --unfreeze_encoder_blocks 0：encoder 與 decoder（simple_fpn/
#       pixel_decoder/m2f_decoder/text projection）全凍，梯度仍穿過它們回流 adapter；
#       decoder_lr_scale 於此設定下無作用（decoder param group 為空）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# e2e 預訓練完整模型（Trainer checkpoint dict，含 model_state_dict）。builder 以 shape-tolerant
# 過濾載入：encoder / pixel_decoder / m2f_decoder / condition_encoder 皆匹配載入，
# adapter 若無匹配鍵則從零初始化（vgg_injector 於 e2e 未訓練，等同全新起步）。
E2E_CKPT="/home/rvl1421/SAM_research-1/segment-anything/outputs_m2f_cs_e2e/weather_sam_best_latest.pth"

# 輸出資料夾：沿用既有慣例（扁平放在 segment-anything/ 根層、outputs_ 前綴，
# 如 outputs_ablation / outputs_recipe / outputs_darkzurich）。
# checkpoint、ablation_config.json、debug 視覺化皆寫入此處。
# 之後傳 --output_dir 仍可臨時覆蓋（"$@" 在最後，argparse 取最後值）。
# 註：本階段為 PEFT 天氣適應，另立新目錄，保留舊的 outputs_m2f_cityscapes 不覆蓋。
OUTPUT_DIR="outputs_m2f_weatheradapt"

# 直接呼叫 python（假設已 conda activate sam_env，此為使用者預設環境）。
# python -u = 無緩衝輸出，啟動與進度條即時顯示。
# 註：不用 conda run，因其會緩衝 stdout，啟動 30-60 秒畫面全空看似「卡住」。
python -u train.py \
  --model_type vit_h \
  --decoder m2f \
  --checkpoint "$E2E_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --use_vgg_adapter \
  --freeze_decoder \
  --unfreeze_encoder_blocks 0 \
  --weight_decay 0.05 \
  --dice_weight 5.0 \
  --cls_weight 2.0 \
  --bce_weight 5.0 \
  --no_object_weight 0.1 \
  --num_points 12544 \
  --lr 5e-5 \
  --max_norm 1.0 \
  --warmup_epochs 5 \
  --warmup_gate_epochs 0 \
  --lovasz_weight 0.0 \
  --m2f_label_smooth \
  --label_smoothing 0.05 \
  "$@"
# --freeze_decoder：凍結 e2e 的 encoder+decoder，只訓 adapter + condition_encoder（PEFT 天氣適應）
# --unfreeze_encoder_blocks 0：encoder 完全凍結（明確標示，防未來 default 變動）
# warmup_epochs=5：保留 LR 暖身（從零訓練 MSDeformAttn/masked-attn decoder 需要，
#                  移除 gate warmup 後更是唯一早期穩定保護）
# warmup_gate_epochs=0：移除 gate warmup（adapter 閘門 epoch 0 起即可訓練）
