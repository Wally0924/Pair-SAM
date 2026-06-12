# 消融實驗執行 Runbook（第 4.9 節）

> 對應 `docs/superpowers/specs/2026-06-01-ablation-experiment-design.md`。
> 目標：10 unique config / 12 訓練 run → 3 張表（累積 / adapter / loss）。
> FULL = R7（MFB-only，無 RCS）；僅 R7(FULL) 跑 3 seeds；C2(取消MFB) = R6 複用。
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
- [x] **磁碟空間**：`df -h .` —— train.py 只保留單一最佳權重（覆寫 `weather_sam_best_latest.pth`，約 3.1G/run），12 runs ≈ **37GB**，無需事後清理。1.2T 剩餘充足。 ✅
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
  --seed 42 --output_dir /tmp/smoke_full
```
> FULL = R7（MFB-only，無 `--rcs`）。
- [ ] **確認 config 落地**：`cat /tmp/smoke_full/ablation_config.json`
      應含 `"inject":"pre","decoder":"unified","lrh":true,"mfb":true,"ref":true,"rcs":false` + seed/loss 權重。
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

**目的**：確認重訓的 FULL（=R7）pipeline 健康、分數非退化、各類別 IoU 合理。**不以 E27 的 65.68% 為硬門檻** —— 該數字為 5/14 舊架構（含 focal 等差異）的單一 seed，不可比；本關卡看的是「能跑通、val mIoU 達現行架構合理量級、長尾類別未崩」。

- [ ] **跑 FULL = R7（seed 42，MFB-only，無 RCS）**，直接寫進正式輸出目錄：
```bash
mkdir -p outputs_ablation
python train.py \
  --epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
  --inject pre --decoder unified --lrh --mfb --lovasz_weight 1 --dice_weight 1 \
  --seed 42 --output_dir outputs_ablation/R7_seed42
```
> FULL = R7（MFB-only，無 `--rcs`）。Run dir 命名為 `R7_seed42`；彙整器自動將 FULL 欄位映射至 R7。
> 不要接 `| tee`：管線會讓 tqdm 進度條退化成每步印一行（洗版）。train.py 已自動寫 `train_log.csv` + `training_curve.png`，無需另存 log。
> 若要背景執行：`nohup python train.py ... > /dev/null 2>&1 &`，再看 `train_log.csv` / `training_curve.png`（背景時進度條本就不顯示）。
- [ ] **監看**：訓練曲線 `outputs_ablation/R7_seed42/training_curve.png`；`train_log.csv` 每 epoch 的 val mIoU。
- [ ] **關卡判定**：訓練結束（early stopping 或 50 epoch）後，best val mIoU 是否落在**現行架構的合理水準**？
      （參考：MFB-only 已得 **67.26%**；重點是管線健康、分數非退化、各類別 IoU 合理。建議跑滿 FULL 3 seeds 看 mean±std 再定錨。）
      - ✅ 達標 → 進入 Phase 3。
      - ❌ 明顯偏低（如 < 65%）→ **停**，代表 pipeline 退化（資料、seed、開關預設值有問題），先排查再續跑，避免浪費 11 個 run 的算力。

---

## Phase 3 — 其餘 11 個 run（算力主體，數天）

R7_seed42（FULL）已在 Phase 2 完成。剩下 11 個。**建議順序：先驗端點，再補中間**（早期發現問題）。

### 3a. 端點與控制組（先跑）
- [ ] **A2 ×1 seed=42**（移除 reference，中心論點控制組）
- [ ] **R7 ×2 剩餘 seeds**（1234, 2026）

### 3b. 累積中間列（單 seed=42，全部無 RCS）
- [ ] R1（baseline）、R2（後置注入）、R3（前置）、R4（統一查詢）、R5（+LRH）、R6（+Lovász/Dice）

### 3c. leave-one-out 變體（單 seed=42）
- [ ] A1（後置注入）、C1（純 CE）
- [ ] **C2 = R6（同 config 複用，無需新 run）**

> **分批捷徑**：12 個 run 的精確指令拆在兩個冪等 script（已完成的 run 自動略過，不重跑）：
> - **Batch 1**：`bash run_ablation_batch1.sh` —— A2/R1/R6/C1（端點與控制組優先；跑完即可先看 loss 表與 A2 落差）
> - **Batch 2**：`bash run_ablation_batch2.sh` —— R2/R3/R4/R5/A1，結束自動全量 eval + 彙整 3 張表

**背景執行建議**（單一 run）：
```bash
nohup python train.py <flags> \
  --output_dir outputs_ablation/<RunID>_seed<N> \
  > outputs_ablation/<RunID>_seed<N>.log 2>&1 &
```
（單卡請**序列執行**，勿同時多個 run 搶 VRAM；有第二張卡才平行。）

---

## Phase 4 — 逐 run 評估

若用 batch script（`run_ablation_batch1.sh` / `run_ablation_batch2.sh`），最後已自動 eval + 彙整，跳到 Phase 6 檢查。

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
      `% tab:ablation_summary`（R1–R7/FULL，7 列）、`% tab:adapter_ablation`（FULL/A1/A2）、`% tab:loss_ablation`（FULL/C1/C2，C2=R6 複用）三段；FULL(R7) 顯示 `mean±std`（3-seed），FULL 的 Δ 為 `---`。

---

## Phase 6 — 數據健全性與誠實性檢查（重要）

- [ ] **FULL(R7) 水準**：R7 三 seed 平均落在合理量級（參考 MFB-only 67.26%）。
- [ ] **累積趨勢**：R1 < R2 < ... < R6 < R7(FULL) 大致遞增？若某步（尤其 R4→R5 LRH、R6→R7 MFB）增益極小或為負 —— **據實寫進正文，不調數據**（spec §5.4）。
- [ ] **A2 落差**：FULL(R7) − A2 是否明顯（中心論點「reference 才是主貢獻」）？
- [ ] **長尾證據**：loss 表的 rider/moto/bike IoU 從 C2(=R6, no-MFB)→FULL(R7) 是否上升（MFB 效果）？
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
| 關卡 | FULL(R7) seed42 完整訓練 | val mIoU ≈ 67.26% |
| 主體 | `run_ablation_batch1.sh` → `run_ablation_batch2.sh` 跑滿 12 run | 各 run 出 `weather_sam_best_latest.pth` + `ablation_config.json` |
| eval | `eval_e1_acdc_val_full.py --ckpt ... --out .../e1_results.json` | 每 run 出 JSON |
| 彙整 | `aggregate_ablation.py --runs_root outputs_ablation` | `ablation_tables.tex`（3 表） |
| 改寫 | 依 paper-rewrite 指引 | 填數值、刪 decoder 表 |
