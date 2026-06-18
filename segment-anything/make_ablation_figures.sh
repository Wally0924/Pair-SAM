#!/usr/bin/env bash
# 產生消融實驗圖（僅 Batch 1 真實數據，無捏造）：
#   1. fig_ablation_batch1_contrib.png — Batch 1 總覽 + 組件貢獻（純 CPU，秒級）
#   2. fig_ablation_allclass.png       — 全 19 類別柱狀圖（純 CPU，秒級）
#   3. fig_ablation_qualitative.png    — Input/GT/R1/A2/FULL 定性比較（需 GPU，數分鐘）
#
# 累積瀑布圖（plot_ablation_waterfall.py）需 R1–R7 全到齊才誠實，
# 缺 R2–R5（Batch 2）時會自動跳過，避免錯誤歸因；Batch 2 後重跑本腳本即啟用。
#
# 全部輸出到 outputs_ablation/。需先 conda activate sam_env。
# 圖 1/2 不需 GPU；圖 3 逐一載入變體推論，單卡 24GB 可行。
set -uo pipefail
cd "$(dirname "$0")"

echo "── [1/4] Batch 1 總覽 + 組件貢獻 ──"
python scripts/eval/plot_ablation_batch1_contrib.py || echo "⚠️ 總覽圖失敗"

echo "── [2/4] 全類別柱狀圖 ──"
python scripts/eval/plot_ablation_allclass.py || echo "⚠️ 柱狀圖失敗"

echo "── [3/4] 累積瀑布圖（缺 Batch 2 會自動跳過）──"
python scripts/eval/plot_ablation_waterfall.py || true

echo "── [4/4] 定性比較圖（GPU 推論，請稍候）──"
python scripts/eval/viz_ablation_compare.py || echo "⚠️ 定性圖失敗"

echo
echo "✅ 完成。輸出："
ls -la outputs_ablation/fig_ablation_*.png 2>/dev/null || echo "（無圖產生，檢查上方錯誤）"
