#!/usr/bin/env bash
# M2F decoder 主線訓練：FPN 直入 decoder、text-init queries、condition token、full M2F loss
set -euo pipefail
cd "$(dirname "$0")/.."

conda run -n sam_env python train.py \
  --decoder m2f \
  --dice_weight 5.0 \
  --decoder_lr_scale 1.0 \
  --lovasz_weight 0.0 \
  --use_vgg_adapter \
  "$@"
# dice_weight 5.0 = M2F 官方值（legacy 預設 1.0，故此處顯式覆蓋）
# decoder_lr_scale 1.0：新 decoder 從零訓練，不需要 legacy 的 0.5 保護縮放
# lovasz_weight 0.0：m2f 路徑不經 ContextLoss，設 0 避免誤導 CSV 讀者
