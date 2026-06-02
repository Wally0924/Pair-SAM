# 消融實驗執行 Runbook（第 4.9 節）

> 對應 `docs/superpowers/specs/2026-06-01-ablation-experiment-design.md`。
> 目標：12 unique config / 14 訓練 run → 3 張表（累積 / adapter / loss）。
> R1–R7 使用 `--no-rcs`；R8(=FULL, +RCS) 與 A1/A2/C1/C2 使用 `--rcs`；僅 R8(FULL) 跑 3 seeds；C2 為獨立 run（mfb off, rcs on）。
> 全程於 `segment-anything/` 目錄、`sam_env` conda 環境執行。

---

## Phase 0 — 起跑前檢查（5 分鐘）

- [x] **環境**：確認當前 `python` 為含齊套件的環境（如已 `conda activate sam_env`）；本 runbook 指令直接使用 `python`，不再前綴 conda ✅ sam_env
- [x] **工作目錄**：`cd /home/rvl1421/SAM_research-1/segment-anything` ✅
- [x] **GPU 空閒**：`nvidia-smi`（確認 24GB 幾乎全空；單 run 需大量 VRAM） ✅ 用 1.1GB/24.5GB、5%
- [x] **資料 CSV 存在**：
      `ls -la /home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_{train,val}.csv` ✅
- [x] **SAM 權重存在**：`ls -lah checkpoints/sam_vit_h_4b8939.pth`（2.4G） ✅
- [x] **CMA 權重存在**：`ls -lah checkpoints/cma_alignment_weights.pth`（69M） ✅
- [x] **class_presence.json**：RCS 採樣前置資料（冪等，可重算）：
      `python scripts/precompute_class_presence.py --csv /home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_train.csv --out /home/rvl1421/SAM_research-1/Datasets/class_presence.json` ✅ 已產出（232K；rider 最稀有 632K px）
- [x] **磁碟空間**：`df -h .` —— train.py 只保留單一最佳權重（覆寫 `weather_sam_best_latest.pth`，約 3.1G/run），14 runs ≈ **44GB**，無需事後清理。1.2T 剩餘充足。 ✅
- [x] **單元測試綠燈**（確認程式碼完好）：
      `python -m pytest segment-anything/tests/ -q`
      實測 **50 passed**（2 warnings，無關）。

---

## Phase 1 — 管線 smoke（1 epoch，約 10–20 分鐘）⭐ 先做這個

**目的**：用最低成本驗證整條管線（建模 → 訓練 → 存檔 → config 落地 → eval 重建 → 彙整）能跑通，再投入數天算力。

- [ ] **跑 1 epoch 的 FULL 到拋棄式目錄**：
```bash
python train.py \
  --epochs 1 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
  --inject pre --decoder unified --lrh --mfb --lovasz_weight 1 --dice_weight 1 \
  --rcs --seed 42 --output_dir /tmp/smoke_full
```
> 此 smoke 含 `--rcs`（= R8/FULL 配置），故需先完成 Phase 0 的 precompute 產出 `class_presence.json`。
- [ ] **確認 config 落地**：`cat /tmp/smoke_full/ablation_config.json`
      應含 `"inject":"pre","decoder":"unified","lrh":true,"mfb":true,"ref":true,"rcs":true` + seed/loss 權重。
- [ ] **確認 RCS 啟用**：訓練 log 開頭應印 `[RCS] enabled (T=0.01); top-5 sampled classes = ...`（含 rider/moto/bike）。
- [ ] **確認 checkpoint 產生**：`ls /tmp/smoke_full/weather_sam_best_latest.pth`
- [ ] **確認 eval 能依 config 重建並算分**：
```bash
python scripts/eval/eval_e1_acdc_val_full.py \
  --ckpt /tmp/smoke_full/weather_sam_best_latest.pth \
  --out /tmp/smoke_full/e1_results.json
cat /tmp/smoke_full/e1_results.json | python -m json.tool | head -20
```
      應看到 `overall_miou` / `per_condition_miou` / `per_class_iou_overall`（1 epoch 分數很低是正常的，這步只驗管線）。
- [ ] **確認彙整器能讀**（單一 run 也能跑，缺的表會印 missing 註解）：
```bash
python scripts/aggregate_ablation.py \
  --runs_root /tmp --results_filename e1_results.json --out /tmp/smoke_tables.tex || true
```
- [ ] 通過後刪除 smoke：`rm -rf /tmp/smoke_full /tmp/smoke_tables.tex`

> ❌ 若任一步失敗 → 先除錯，**不要**進入 Phase 2。

---

## Phase 2 — Pipeline 關卡：FULL（最多 50 epoch，早停約 30–40，約半天）⭐ 第二個做

> 註：`--epochs` 同時是 cosine LR 衰減的時程。50 比 80 衰減更陡（同 epoch 的 LR 更低），與 5/14 E27（≈80 時程）較不可比；這是刻意選擇。

**目的**：確認重訓的 FULL（=R8）pipeline 健康、分數非退化、各類別 IoU 合理。**不以 E27 的 65.68% 為硬門檻** —— 該數字為 5/14 舊架構（含 focal 等差異）的單一 seed，不可比；本關卡看的是「能跑通、val mIoU 達現行架構合理量級、長尾類別未崩」。

- [ ] **跑 FULL = R8（seed 42，含 RCS）**，直接寫進正式輸出目錄：
```bash
mkdir -p outputs_ablation
python train.py \
  --epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
  --inject pre --decoder unified --lrh --mfb --lovasz_weight 1 --dice_weight 1 \
  --rcs --seed 42 --output_dir outputs_ablation/R8_seed42
```
> FULL = R8 = +RCS（最後累積步驟）。Run dir 命名為 `R8_seed42`；彙整器自動將 FULL 欄位映射至 R8。
> 不要接 `| tee`：管線會讓 tqdm 進度條退化成每步印一行（洗版）。train.py 已自動寫 `train_log.csv` + `training_curve.png`，無需另存 log。
> 若要背景執行：`nohup python train.py ... > /dev/null 2>&1 &`，再看 `train_log.csv` / `training_curve.png`（背景時進度條本就不顯示）。
- [ ] **監看**：訓練曲線 `outputs_ablation/R8_seed42/training_curve.png`；`train_log.csv` 每 epoch 的 val mIoU。
- [ ] **關卡判定**：訓練結束（early stopping 或 50 epoch）後，best val mIoU 是否落在**現行架構的合理水準**？
      （參考：seed42 FULL 已得 **63.47%**；5/14 的 65.68% 是舊架構，不必硬追。重點是管線健康、分數非退化、各類別 IoU 合理。建議跑滿 FULL 3 seeds 看 mean±std 再定錨。）
      - ✅ 達標 → 進入 Phase 3。
      - ❌ 明顯偏低（如 < 64%）→ **停**，代表 pipeline 退化（資料、seed、開關預設值有問題），先排查再續跑，避免浪費 15 個 run 的算力。

---

## Phase 3 — 其餘 13 個 run（算力主體，數天）

R8_seed42（FULL）已在 Phase 2 完成。剩下 13 個。**建議順序：先驗端點，再補中間**（早期發現問題）。

### 3a. 端點與控制組（先跑）
- [ ] **A2 ×1 seed=42**（移除 reference，中心論點控制組；rcs on）
- [ ] **R7 ×1 seed=42**（= 舊 FULL，--no-rcs，RCS leave-one-out 控制組）
- [ ] **R8 ×2 剩餘 seeds**（1234, 2026）

### 3b. 累積中間列（單 seed=42，全部 --no-rcs）
- [ ] R1（baseline）、R2（後置注入）、R3（前置）、R4（統一查詢）、R5（+LRH）、R6（+Lovász/Dice）

### 3c. leave-one-out 變體（單 seed=42，全部 --rcs）
- [ ] A1（後置注入）、C1（純 CE）、C2（取消 MFB，獨立 run，mfb off rcs on）

> **一次跑完全部的捷徑**：上述 14 個 run 的精確指令都在 `run_ablation.sh`（含 Step 0 precompute）。若 Phase 2 已單獨跑了 R8_seed42，直接整檔執行會重跑它（覆寫、浪費一次）。建議二選一：
> - **(A) 逐行貼**：打開 `run_ablation.sh`，跳過 R8_seed42 那行，其餘逐一/分批貼到終端機（可背景跑）。
> - **(B) 整檔跑**：先 `rm -rf outputs_ablation/R8_seed42`，再 `bash run_ablation.sh`（讓它從頭一致地跑完 14 個，含 eval + 彙整）。

**背景執行建議**（單一 run）：
```bash
nohup python train.py <flags> \
  --output_dir outputs_ablation/<RunID>_seed<N> \
  > outputs_ablation/<RunID>_seed<N>.log 2>&1 &
```
（單卡請**序列執行**，勿同時多個 run 搶 VRAM；有第二張卡才平行。）

---

## Phase 4 — 逐 run 評估

若用 `bash run_ablation.sh`（捷徑 B），最後已自動 eval + 彙整，跳到 Phase 6 檢查。

若手動逐行跑（捷徑 A），對每個完成的 run 執行：
- [ ] ```bash
      for d in outputs_ablation/*/; do
        ckpt="$d/weather_sam_best_latest.pth"
        [ -f "$ckpt" ] && python scripts/eval/eval_e1_acdc_val_full.py \
          --ckpt "$ckpt" --out "$d/e1_results.json"
      done
      ```
      eval 會依各 run 的 `ablation_config.json` **自動重建相同配置**（decoder/lrh/ref 一致）。

---

## Phase 5 — 彙整 3 張表

- [ ] ```bash
      python scripts/aggregate_ablation.py \
        --runs_root outputs_ablation --out outputs_ablation/ablation_tables.tex
      ```
- [ ] 檢視 `outputs_ablation/ablation_tables.tex`：應含
      `% tab:ablation_summary`（R1–R8/FULL，8 列含 R7 控制組）、`% tab:adapter_ablation`（FULL/A1/A2）、`% tab:loss_ablation`（FULL/C1/C2）三段；FULL(R8) 顯示 `mean±std`（3-seed），FULL 的 Δ 為 `---`。

---

## Phase 6 — 數據健全性與誠實性檢查（重要）

- [ ] **FULL(R8) ≈ E27**：R8 三 seed 平均落在 65.68% 量級。
- [ ] **累積趨勢**：R1 < R2 < ... < R7 < R8(FULL) 大致遞增？若某步（尤其 R4→R5 LRH、R6→R7 MFB、R7→R8 RCS）增益極小或為負 —— **據實寫進正文，不調數據**（spec §5.4）。
- [ ] **RCS 長尾效益**：R7→R8 的 bus/moto/bicycle IoU 是否提升（支持 RCS 論據）？
- [ ] **A2 落差**：FULL(R8) − A2 是否明顯（中心論點「reference 才是主貢獻」）？
- [ ] **長尾證據**：loss 表的 rider/moto/bike IoU 從 C2→FULL(R8) 是否上升（MFB 效果）？
- [ ] **LRH 誠實標註**：未做 Boundary metric，LRH 僅以整體 mIoU 之有限增益陳述，**不得宣稱未量測的 boundary 數據**。

---

## Phase 7 — 改寫論文 4.9 節

- [ ] 依 `docs/superpowers/specs/2026-06-01-paper-rewrite-4.9-ablation.md` 操作：
      只刪 decoder 表（4.9.2）、保留 adapter/loss 表、把被刪論述折進累積表、搬移 §1.2.1 / §1.2.2 交叉引用、節號遞補。
- [ ] 把 `ablation_tables.tex` 的數值填入三張表的 `[XX]` 欄位。
- [ ] ACDC 類別像素頻率（rider/moto/bike `[X.XXX]%`）從 `utils/new_loss.py` 的 `_ACDC_CLASS_FREQ` 導出填正文。

---

## 一頁速查

| 步驟 | 指令重點 | 預期 |
|------|---------|------|
| smoke | `train.py --epochs 1 ... --output_dir /tmp/smoke_full` + eval + aggregate | 管線跑通 |
| 關卡 | FULL seed42 完整訓練 | val mIoU ≈ 65.68% |
| 主體 | `run_ablation.sh`（或逐行）跑滿 14 run | 各 run 出 `weather_sam_best_latest.pth` + `ablation_config.json` |
| eval | `eval_e1_acdc_val_full.py --ckpt ... --out .../e1_results.json` | 每 run 出 JSON |
| 彙整 | `aggregate_ablation.py --runs_root outputs_ablation` | `ablation_tables.tex`（3 表） |
| 改寫 | 依 paper-rewrite 指引 | 填數值、刪 decoder 表 |
