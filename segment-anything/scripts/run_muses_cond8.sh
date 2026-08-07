#!/usr/bin/env bash
# =============================================================================
# PairSAM MUSES 訓練 + 驗證(8 條件:weather × time_of_day 全交叉)
#
# 目的:
#   在 MUSES train(1500 張)上做 in-domain 監督訓練,使結果可與 MUSES 論文
#   Table 4 的 7 個架構直接排名比較
#   (見 docs/experiments/2026-08-03-muses-comparison-framework.md)。
#
# 起始權重見下方 INIT 變數。預設 cs(Cityscapes e2e 預訓練)與 ACDC 實驗走
# 完全相同的兩階段管線,只是把第二階段的惡劣天氣資料集換成 MUSES。
#
# 結構差異:
#   --num_conditions 8:MUSES 是 weather × time_of_day 的二維交叉,ACDC 僅一維。
#     編號設計使前 4 格語意與 ACDC 相同(fog/day=0, rain/day=1, snow/day=2,
#     clear/night=3),故 builder 可把 4×256 condition embedding 原位沿用,新增的
#     4 格(fog/night=4, rain/night=5, snow/night=6, clear/day=7)由語意相近的
#     舊列平均初始化。詳見 build_pair_sam._expand_condition_embedding。
#     註:INIT=cs 時來源的 condition embedding 本身是退化的(CS 預訓練 cond=false,
#     只有索引 0 收過梯度),擴表照樣執行但不攜帶天氣先驗——這是預期行為。
#
# 超參來源:outputs_ablation_m2f/FULL_seed42/ablation_config.json,逐項對齊。
#   ★ 其中 6 項與 train.py 的 argparse 預設不同,必須顯式傳入,否則會變成
#     另一組實驗而無法與論文中的 Pair-SAM 比較:
#       dice_weight 5.0(預設 1.0)   lovasz_weight 0.0(預設 1.0)
#       weight_decay 0.05(預設 1e-2) decoder_lr_scale 1.0(預設 0.5)
#       warmup_gate_epochs 0(預設 3) --m2f_label_smooth(預設關閉)
#
# 用法(比照 ablation_m2f_phase*.sh:腳本不自行 tee,由呼叫端決定去向):
#   # 前景跑,進度條正常滾動
#   bash scripts/run_muses_cond8.sh
#
#   # 背景跑並留檔
#   nohup bash scripts/run_muses_cond8.sh > outputs_muses_cond8/run.log 2>&1 &
#
#   INIT=acdc OUTPUT_DIR=outputs_muses_cond8_acdc \
#       bash scripts/run_muses_cond8.sh                 # 改由 ACDC 權重起訓
#   SKIP_TRAIN=1 bash scripts/run_muses_cond8.sh        # 只跑驗證(權重已存在時)
#   SKIP_TEST=1  bash scripts/run_muses_cond8.sh        # 訓練 + val,不產生提交檔
#
# nohup 留下的 run.log 會含 tqdm 的每次刷新。事後整理成每 epoch 一條:
#   tr '\r' '\n' < outputs_muses_cond8/run.log \
#     | awk '!/%\|/ || /100%\|/' > outputs_muses_cond8/run.tidy.log
# 執行中只看摘要:
#   grep -a "Val mIoU\|New best" outputs_muses_cond8/run.log
#
# 產物:
#   ${OUTPUT_DIR}/weather_sam_best_latest.pth   最佳權重
#   ${OUTPUT_DIR}/ablation_config.json          eval 重建模型用(含 num_conditions)
#   ${OUTPUT_DIR}/train_log.csv                 逐 epoch 指標(trainer 自行寫出)
#   submissions/muses_test_cond8/               test 預測 PNG
#   submissions/muses_test_cond8.zip            MUSES evaluation server 提交檔
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/rvl1421/SAM_research-1"
SEGANY="${REPO_ROOT}/segment-anything"
cd "${SEGANY}"

# shellcheck disable=SC1091
source /home/rvl1421/miniconda3/etc/profile.d/conda.sh
conda activate sam_env

OUTPUT_DIR="${OUTPUT_DIR:-outputs_muses_cond8}"

# 起始權重(INIT= 環境變數切換):
#   cs  (預設) = Cityscapes e2e 預訓練。與 ACDC 實驗走完全相同的兩階段管線
#                (CS 預訓練 → 惡劣天氣資料集適應),adapter 於 MUSES 1500 張上
#                從零學起,與 ACDC 的 1600 張規模相當。額外監督資料只有
#                Cityscapes,與論文 Table 4 的比較最站得住腳。
#   acdc       = ACDC 訓練完成的 Pair-SAM。adapter 與 condition 已訓練,分數
#                預期較高,但額外用掉 1600 張惡劣天氣標註,對 Table 4 不對等。
case "${INIT:-cs}" in
  cs)   INIT_CKPT="${SEGANY}/outputs_m2f_cs_e2e/weather_sam_best_latest.pth" ;;
  acdc) INIT_CKPT="${SEGANY}/outputs_ablation_m2f/FULL_seed42/weather_sam_best_latest.pth" ;;
  *)    echo "❌ INIT 只接受 cs 或 acdc(收到:${INIT})" >&2; exit 1 ;;
esac
echo "起始權重(INIT=${INIT:-cs}):${INIT_CKPT}"
[ -f "${INIT_CKPT}" ] || { echo "❌ 找不到起始權重" >&2; exit 1; }

BEST_CKPT="${OUTPUT_DIR}/weather_sam_best_latest.pth"

TRAIN_CSV="${REPO_ROOT}/Datasets/muses_cond8_ref_rgb_train.csv"
VAL_CSV="${REPO_ROOT}/Datasets/muses_cond8_ref_rgb_val.csv"
TEST_CSV="${REPO_ROOT}/Datasets/muses_cond8_ref_rgb_test.csv"

mkdir -p "${OUTPUT_DIR}"

# ── Stage 0:準備 CSV ────────────────────────────────────────────────
# map8 = 8 類全交叉映射;--verify-paths 逐列確認影像/參考/GT 檔案存在,
# 避免訓練跑到一半才因缺檔中斷。已存在則跳過(冪等)。
echo "═══ Stage 0:準備 CSV ═══"
for f in "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}"; do
  if [ ! -f "${f}" ]; then
    echo "缺少 $(basename "${f}") — 重新產生全部 split"
    python "${REPO_ROOT}/Datasets/make_muses_csv.py" \
      --splits train val test --cond-mode map8 \
      --out-prefix muses_cond8_ref_rgb --verify-paths
    break
  fi
done
wc -l "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}"

# ── Stage 0b:起始權重前置檢查(約 10 秒)─────────────────────────────
# builder 的 state_dict 過濾會「靜默」丟棄 shape 不符的鍵:若 4→8 擴表未生效,
# condition_encoder 會整張退回隨機初始化,訓練照跑不報錯,只是分數莫名偏低。
# 在此先建一次模型驗證,壞掉就立刻停,不浪費 9 小時。
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo
  echo "═══ Stage 0b:起始權重前置檢查 ═══"
  INIT_CKPT="${INIT_CKPT}" python -u - <<'PYEOF'
import io, os, contextlib, torch
from segment_anything.build_pair_sam import build_pair_sam_from_config, _COND_KEY

ckpt = os.environ['INIT_CKPT']
raw = torch.load(ckpt, map_location='cpu', weights_only=False)
raw = raw.get('model_state_dict', raw)
src = raw[_COND_KEY]
assert src.shape[0] == 4, f'起始權重的 condition_encoder 應為 4 列,實得 {src.shape[0]}'

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m = build_pair_sam_from_config(
        dict(model_type='vit_h', use_vgg_adapter=True, inject='pre', decoder='m2f',
             lrh=False, mfb=False, ref=True, cond=True,
             adapter_variant='reference', num_conditions=8),
        checkpoint=ckpt)
log = buf.getvalue()
assert '擴表 4 → 8' in log, '擴表未觸發'

got = m.condition_encoder.weight.detach()
assert got.shape == (8, 256), f'condition_encoder 形狀錯誤:{tuple(got.shape)}'
assert torch.equal(got[:4], src), '前 4 列未逐位元沿用起始權重'
print(f'✅ condition_encoder 4 → 8 擴表正確,前 4 列逐位元一致')
print('  ' + next(l for l in log.splitlines() if 'Missing keys' in l))
PYEOF
fi

# ── Stage 1:訓練 ────────────────────────────────────────────────────
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo
  echo "═══ Stage 1:訓練(1500 張 / 30 epochs,約 9 小時)═══"
  python -u train.py \
    --train_csv "${TRAIN_CSV}" \
    --val_csv   "${VAL_CSV}" \
    --num_conditions 8 \
    --checkpoint "${INIT_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 --epochs 30 --patience 10 \
    --lr 5e-5 --warmup_epochs 5 --weight_decay 0.05 --max_norm 1.0 \
    --decoder m2f --adapter_variant reference --cond --ref \
    --dice_weight 5.0 --lovasz_weight 0.0 \
    --decoder_lr_scale 1.0 --warmup_gate_epochs 0 \
    --encoder_lr_scale 0.005 --unfreeze_encoder_blocks 0 \
    --cls_weight 2.0 --bce_weight 5.0 --no_object_weight 0.1 --num_points 12544 \
    --label_smoothing 0.05 --m2f_label_smooth
else
  echo "═══ Stage 1:略過(SKIP_TRAIN=1)═══"
fi

[ -f "${BEST_CKPT}" ] || { echo "❌ 找不到權重 ${BEST_CKPT}" >&2; exit 1; }

# ── Stage 2:val 診斷(250 張)──────────────────────────────────────
# ⚠️ val 已在訓練中用於挑選 best checkpoint(argmax val mIoU + patience 早停),
#    因此本階段的數字帶有選擇偏誤,**不是獨立效能估計,不可作為 headline,
#    也不可與 MUSES 論文 Table 4/5 對照(那些全是 test 分數)**。
#
#    本階段的兩個正當用途:
#    (1) 管線驗證:確認原生解析度推論路徑正常,再花力氣產生 test 提交檔。
#        (2026-07-14 零樣本評估即用此法:val 63.39 vs test 62.49,差 0.90 → 管線正確)
#    (2) 分條件診斷:訓練時的 val 是 1024×1024 confusion matrix、且只印整體與逐類別;
#        本腳本則是 logits 上採至原生 1080×1920,並額外給出分天氣 / 分日夜細分。
#
# --cond-mode csv:沿用 CSV 既有的 8 類 condition_id,不做任何重新映射。
#   (map 模式是 4 類 ACDC 映射,套在 8 條件模型上會取到錯誤的 embedding 列。)
echo
echo "═══ Stage 2:val 診斷(選擇偏誤資料,僅供管線驗證與分條件分析)═══"
python -u scripts/eval/dump_muses_preds.py \
  --split val --cond-mode csv \
  --csv "${VAL_CSV}" --ckpt "${BEST_CKPT}"

# ── Stage 3:test 提交檔 ────────────────────────────────────────────
# MUSES test 的 GT 未公開,只能經官方 evaluation server 評分 ——
# 這也使 test 成為**唯一未被選擇程序污染**的分數,是可與論文對照的那一個。
if [ "${SKIP_TEST:-0}" != "1" ]; then
  echo
  echo "═══ Stage 3:test 預測 + 打包 ═══"
  # --out 指向獨立目錄:submissions/muses_test/ 是 2026-07-14 的零樣本提交檔
  # (對應 test mIoU 62.49),run_test 會清空目標目錄的既有 PNG,不可覆寫。
  python -u scripts/eval/dump_muses_preds.py \
    --split test --cond-mode csv \
    --csv "${TEST_CSV}" --ckpt "${BEST_CKPT}" --zip \
    --out "${SEGANY}/submissions/muses_test_cond8"
  echo
  echo "提交檔:${SEGANY}/submissions/muses_test_cond8.zip"
  echo "(零樣本舊檔 submissions/muses_test.zip 保持不動)"
  echo "上傳至 https://muses.vision.ee.ethz.ch/benchmarks(semantic segmentation)"
else
  echo "═══ Stage 3:略過(SKIP_TEST=1)═══"
fi

echo
echo "═══ 完成 ═══"
echo "逐 epoch 指標:${OUTPUT_DIR}/train_log.csv"
echo "val 診斷結果 :見上方 Stage 2 輸出"
echo "           ⚠️ 選擇偏誤資料(best ckpt 由它挑出)。只用於管線驗證與分條件分析,"
echo "              不可作為 headline,也不可與論文 Table 4/5 對照。"
echo
echo "可對照的分數:上傳 submissions/muses_test_cond8.zip 後由 evaluation server 給出的 test mIoU。"
echo "對照基準    :MUSES 論文 Table 4(皆為 test,in-domain 監督,與本 run 對等)"
echo "              DeepLabv3+ 70.5 / SegFormer-B5 74.7 / Mask2Former-Swin-L 77.1"
echo "              見 docs/experiments/2026-08-03-muses-comparison-framework.md"
