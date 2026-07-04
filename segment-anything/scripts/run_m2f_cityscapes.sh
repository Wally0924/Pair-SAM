#!/usr/bin/env bash
# =============================================================================
# WeatherSAM M2F 訓練：SAM ViT-H（Cityscapes fine-tuned encoder）+ MSDeformAttn
# pixel decoder + masked-attention decoder。所有超參已對照原始碼/論文稽核。
#
# 參數來源稽核（2026-07-04）：
#   [M2F 官方 Mask2Former] cls=2.0 / bce=5.0 / dice=5.0 / no_object=0.1 /
#       num_points=12544 / oversample=3.0 / importance=0.75（後三者為 m2f_loss 內建預設）
#   [對齊官方] weight_decay=0.05（Mask2Former SOLVER.WEIGHT_DECAY）
#   [保留現值] max_norm=1.0（官方 0.01 為 ResNet/Swin 全訓練調校，不套用於本凍結設定）
#   [設計依據] lr=5e-5（凍結 backbone fine-tune）/ Warmup+Cosine / eff.batch=4（4090）/
#       decoder_lr_scale=1.0（新 decoder 從零訓練）/ lovasz=0.0（m2f 不走 ContextLoss）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CITYSCAPES_ENC="/home/rvl1421/SAM_research-1/segment-anything/checkpoints/cityscapes_pretrain/sam_vit_h_cityscapes_encoder_best.pth"

# 輸出資料夾：沿用既有慣例（扁平放在 segment-anything/ 根層、outputs_ 前綴，
# 如 outputs_ablation / outputs_recipe / outputs_darkzurich）。
# checkpoint、ablation_config.json、debug 視覺化皆寫入此處。
# 之後傳 --output_dir 仍可臨時覆蓋（"$@" 在最後，argparse 取最後值）。
OUTPUT_DIR="outputs_m2f_cityscapes"

# 直接呼叫 python（假設已 conda activate sam_env，此為使用者預設環境）。
# python -u = 無緩衝輸出，啟動與進度條即時顯示。
# 註：不用 conda run，因其會緩衝 stdout，啟動 30-60 秒畫面全空看似「卡住」。
python -u train.py \
  --model_type vit_h \
  --decoder m2f \
  --checkpoint "$CITYSCAPES_ENC" \
  --output_dir "$OUTPUT_DIR" \
  --use_vgg_adapter \
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
  "$@"
# warmup_epochs=5：保留 LR 暖身（從零訓練 MSDeformAttn/masked-attn decoder 需要，
#                  移除 gate warmup 後更是唯一早期穩定保護）
# warmup_gate_epochs=0：移除 gate warmup（adapter 閘門 epoch 0 起即可訓練）
