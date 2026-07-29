#!/usr/bin/env bash
# =============================================================================
# PairSAM M2F 訓練：凍結 encoder + 解凍 decoder 微調（+ DeformAdapter）
#   來源模型 = outputs_m2f_cs_e2e（Cityscapes 晴天端到端預訓練的 encoder + MSDeformAttn
#   pixel decoder + masked-attention decoder）。本階段載入該完整權重（encoder 與 decoder
#   皆載入預權重），凍結 ViT-H encoder，於 ACDC 惡劣天氣上微調 decoder，並一併訓練
#   DeformAdapter（vgg_injector）+ condition_encoder + text projection。
#
#   與 run_m2f_cityscapes.sh 的差異：該腳本為 --freeze_decoder（PEFT，只訓 adapter+
#   condition）；本腳本移除 freeze_decoder，讓 decoder 一起微調。
#
# 參數來源稽核（2026-07-04）：
#   [M2F 官方 Mask2Former] cls=2.0 / bce=5.0 / dice=5.0 / no_object=0.1 /
#       num_points=12544 / oversample=3.0 / importance=0.75（後三者為 m2f_loss 內建預設）
#   [對齊官方] weight_decay=0.05（Mask2Former SOLVER.WEIGHT_DECAY）
#   [保留現值] max_norm=1.0（官方 0.01 為 ResNet/Swin 全訓練調校，不套用於本凍結設定）
#   [設計依據] lr=5e-5（微調 base LR，adapter 另乘 adapter_lr_scale 預設 3×）/
#       Warmup+Cosine / eff.batch=4（4090）/ lovasz=0.0（m2f 不走 ContextLoss）/
#       decoder_lr_scale=1.0（微調預訓練 decoder，非從零訓練；5e-5 本即微調級 LR）
#   [凍結策略] --unfreeze_encoder_blocks 0：encoder 完全凍結；不加 --freeze_decoder，
#       故 simple_fpn/pixel_decoder/m2f_decoder 皆解凍微調。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# e2e 預訓練完整模型（Trainer checkpoint dict，含 model_state_dict）。builder 以 shape-tolerant
# 過濾載入：encoder / simple_fpn / pixel_decoder / m2f_decoder / condition_encoder 皆匹配載入，
# adapter 若無匹配鍵則從零初始化（vgg_injector 於 e2e 未訓練，等同全新起步）。
E2E_CKPT="/home/rvl1421/SAM_research-1/segment-anything/outputs_m2f_cs_e2e/weather_sam_best_latest.pth"

# 輸出資料夾：另立新目錄，與 outputs_m2f_cityscapes / outputs_m2f_weatheradapt 區隔。
# 之後傳 --output_dir 仍可臨時覆蓋（"$@" 在最後，argparse 取最後值）。
OUTPUT_DIR="outputs_m2f_decoder_ft"

# 直接呼叫 python（假設已 conda activate sam_env，此為使用者預設環境）。
# python -u = 無緩衝輸出，啟動與進度條即時顯示。
python -u train.py \
  --model_type vit_h \
  --decoder m2f \
  --checkpoint "$E2E_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --use_vgg_adapter \
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
  --decoder_lr_scale 1.0 \
  --lovasz_weight 0.0 \
  --m2f_label_smooth \
  --label_smoothing 0.05 \
  "$@"
# --unfreeze_encoder_blocks 0：encoder 完全凍結（明確標示，防未來 default 變動）
# 不加 --freeze_decoder：decoder（simple_fpn/pixel_decoder/m2f_decoder）解凍微調
