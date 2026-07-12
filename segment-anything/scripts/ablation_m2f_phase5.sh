#!/usr/bin/env bash
# =============================================================================
# M-series Phase 5 — 語義B重跑 + 新增子設計消融(約 4 天,接在 Phase 4 之後)
#
# 【重跑】B1/W2 的舊 run 是在 conf=1 修正(commit ccc99d9,2026-07-09)之前
#   訓練的:當時 --no-ref 只把參考特徵 c 歸零、conf 仍取自參考遮罩置信度,
#   經 feat = LayerNorm(c)*conf = bias*conf 洩漏參考的空間訊號,非乾淨的
#   語義B「容量控制」基準。以修正碼重跑,舊 run 保留供對照不覆蓋。
#   B1_semB:碩論 tab:ablation_bseries 第 2 列(+Adapter 無參考無 cond)
#   W2_semB:碩論 tab:ablation_wseries「移除參考影像」列
#   判讀:若 B1_semB < 舊 B1(76.00),即證實舊值被洩漏灌高、B2−B1 參考
#   淨貢獻相應拉開;結果無論方向據實更新論文(規劃 §5 誠實性)。
#
# 【補跑】碩論 tab:ablation_wseries 尚缺的兩個子設計消融(2026-07-11 已加旗標):
#   W3_confmod:--no-conf_mod = 移除置信度調變(m̄≡1,參考全幅注入),
#     檢驗貢獻一的置信度感知子設計。
#   W6_noext:--no-extractor = 移除抽取器(單向注入,參考先驗不隨主幹深度
#     更新;build 於 hook 註冊前清空,無 trainable-but-unused 死參數),
#     檢驗貢獻一的雙向結構子設計。
#   註:編號依 2026-07-09 統整報告/碩論 W 表列序(W3=置信度調變、W6=抽取器);
#      舊規劃「W3=B2 複用(cond 軸)」已由 B 系列與 2×2 交互涵蓋,不再使用。
#
# 【預期警報】B1_semB/W2_semB(--no-ref)首個 backward 會印
#   「[Grad Audit] …未收到梯度:vgg_injector」——良性。斷聯的只有 rpm.proj_c2/c3/c4
#   與 rpm.level_embed(參考特徵翻譯器,c=zeros_like 使其輸出被丟棄,語義B設計預期);
#   injector/extractor 均在計算圖上(gamma 有非零梯度,已以 CPU 小型復現+舊 B1/W2
#   遙測 gate 0.0009→0.0083 雙重驗證,2026-07-12)。稽核為 any-param-None 即報頂層模組。
#
# 假設已 conda activate sam_env。背景執行:
#   nohup bash scripts/ablation_m2f_phase5.sh > outputs_ablation_m2f/phase5.log 2>&1 &
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ablation_m2f_common.sh
mkdir -p "$OUT_ROOT"

# --- 重跑(語義B,優先:C1 參考淨貢獻的樞紐) ---
run_one "B1_semB_seed42" --no-ref --no-cond
run_one "W2_semB_seed42" --no-ref

# --- 補跑(碩論 W 表缺口) ---
run_one "W3_confmod_seed42" --no-conf_mod
run_one "W6_noext_seed42" --no-extractor

echo "===== Phase 5 完成 ====="
print_summary
