#!/usr/bin/env bash
# =============================================================================
# WeatherSAM M2F「Cityscapes 端到端預訓練」(Strategy B)：
#   encoder + m2f decoder 一起訓，產出「兩者皆 co-adapt」的預權重。
#
# 與凍結版 run_m2f_cs_pretrain.sh 的唯一差異：解凍 ViT-H 讓 encoder 與
# m2f decoder 在 Cityscapes 上端到端 co-adapt（ViTDet / Mask2Former 標準做法）。
#
#   --unfreeze_encoder_blocks 32：解凍全部 32 個 ViT-H block（patch_embed/neck
#       仍凍結）。encoder 起點回到「原始 SAM foundation」（SA-1B 預訓練），由 m2f
#       目標直接驅動域內化，不經 Stage-1 CS encoder 中間產物（避免同一資料集透過
#       將被丟棄的臨時頭重複適配的 confound）。「e2e」=拓撲上 encoder+decoder 一起
#       解凍聯訓，非隨機初始化。
#   --encoder_lr_scale 0.1：encoder LR = lr × 0.1 = 5e-6（decoder 的 1/10）。
#       gentle co-adapt，防止災難性遺忘 SAM/CS 特徵。★ 此為主要可調旋鈕：
#       想更保守 → 0.02（≈pretrain_cityscapes 的 1/10）；想更用力 → 0.2。
#
# 目的：得到 encoder+decoder 皆勝任 Cityscapes 任務的完整模型，使 ACDC 階段
#   可「凍結 encoder+decoder，只訓 adapter+condition」（純參數高效天氣適應）。
#
# 產物：outputs_m2f_cs_e2e/weather_sam_best_latest.pth
#   ACDC 微調：--checkpoint outputs_m2f_cs_e2e/weather_sam_best_latest.pth
#     （build 依名稱+形狀匹配，encoder 與 decoder 權重一併載入）
#
# 成本提醒：解凍 encoder → 每步需 backprop 穿 ViT-H，約慢 3-4×。從原始 SAM 起步
#   少了 CS 熱啟動，收斂較慢 → --epochs 30（含 5 warmup）。encoder_lr_scale 0.1 =
#   encoder 5e-6（慢）、decoder 5e-5（快 10×）→ decoder 先學好、encoder 慢調，
#   正好保護 SAM 特徵不被早期隨機 decoder 的雜訊梯度沖壞。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# 起始權重：原始 SAM（SA-1B foundation）。載入器按 key+shape 過濾，只有 image_encoder.*
# 命中覆蓋，其餘（prompt_encoder / legacy mask_decoder）m2f 路徑用不到、無害。
SAM_INIT="/home/rvl1421/SAM_research-1/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
OUTPUT_DIR="outputs_m2f_cs_e2e"

python -u train.py \
  --model_type vit_h \
  --decoder m2f \
  --checkpoint "$SAM_INIT" \
  --output_dir "$OUTPUT_DIR" \
  --train_csv /home/rvl1421/SAM_research-1/Datasets/cityscapes_m2f_train.csv \
  --val_csv /home/rvl1421/SAM_research-1/Datasets/cityscapes_m2f_val.csv \
  --no-use_vgg_adapter \
  --no-ref \
  --no-cond \
  --unfreeze_encoder_blocks 32 \
  --encoder_lr_scale 0.1 \
  --weight_decay 0.05 \
  --dice_weight 5.0 \
  --cls_weight 2.0 \
  --bce_weight 5.0 \
  --no_object_weight 0.1 \
  --num_points 12544 \
  --lr 5e-5 \
  --max_norm 1.0 \
  --epochs 30 \
  --warmup_epochs 5 \
  --warmup_gate_epochs 0 \
  --decoder_lr_scale 1.0 \
  --lovasz_weight 0.0 \
  --m2f_label_smooth \
  --label_smoothing 0.05 \
  "$@"
