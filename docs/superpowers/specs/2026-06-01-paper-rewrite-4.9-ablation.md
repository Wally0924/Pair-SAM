# 論文改寫指引 — 第 4.9 節消融實驗（保留 adapter+loss 表、僅刪 decoder 表）

> 對應 `chapter4-experiment.tex` 第 4.9 節。本文件獨立於實作 spec（`2026-06-01-ablation-experiment-design.md`），專供「數據齊備後改寫論文」時照做。
> **範圍決策**：保留累積表 `tab:ablation_summary`、adapter 表 `tab:adapter_ablation`、loss 表 `tab:loss_ablation`；**僅刪除 decoder 表 `tab:decoder_ablation`（4.9.2）**。
> 撰寫日期：2026-06-01。

---

## 0. 改寫總覽（一句話）

**只動 4.9.2**：刪除解碼端消融小節與 `tab:decoder_ablation`，其有效論述（逐類別解碼之破碎、移除 LRH）折進保留的「4.9.4 消融總結」累積表討論。4.9.1（adapter）、4.9.3（loss）、4.9.4（累積）**維持原結構**，僅按實測數據填 `[XX]` 與微調誠實性敘述。

> 相較先前「刪 3 表」的方案，本範圍對論文改動最小，幾乎只少一張表。

---

## 1. 刪除清單（僅 1 項）

| 刪除對象 | 原位置 | 處理 |
|---|---|---|
| 小節 4.9.2「解碼端消融」 | `\subsection{解碼端消融}` | **刪**。兩段論述搬移（見 §2） |
| 表 `tab:decoder_ablation` | 4.9.2 內 | **刪** |

> ⚠️ **交叉引用搶救**：4.9.2 內含「對應第 1.2.2 節」之呼應（逐類別獨立解碼 → 結構限制論點）。刪小節時此 `\ref` / 章節呼應**必須搬到累積表 R3→R4 討論**（見 §2），否則前章 1.2.2 失去實驗對應。
> 全文 grep `tab:decoder_ablation` 確認無其他章節 `\ref` 它（若有，改指 `tab:ablation_summary`）。

---

## 2. 被刪論述的去處（折入 4.9.4 累積表討論）

4.9.4 累積表 `tab:ablation_summary` 已逐列討論。把 4.9.2 兩段論述折入對應轉換：

| 累積轉換 | 折入的原 4.9.2 論述 | 改寫要點 |
|---|---|---|
| **R3→R4（逐類別→統一查詢）** | 原「逐類別獨立解碼」段 + **第 1.2.2 節呼應** | 統一查詢的跨類別 self-attention 修補自相近／相鄰類別（如 road/sidewalk、person/rider）在惡劣天氣下的破碎輸出；**第 1.2.2 節呼應搬到此處**。 |
| **R4→R5（+LRH）** | 原「移除 LRH」段 | ⚠️ 因**不做 Boundary metric**，僅能以 mIoU 之小幅正增益陳述；**誠實標註**：LRH 主要作用於邊界精修，本文未獨立量測邊界指標，故僅報整體 mIoU 之有限增益（不得宣稱未量測的 boundary 數據）。 |

---

## 3. 保留小節的處理（4.9.1 / 4.9.3 / 4.9.4）

維持原結構與敘事，僅依實測數據填值、微調誠實性。

### 3.1 4.9.1 注入機制（Adapter）消融 — `tab:adapter_ablation`【保留】
- 三列維持：完整模型（前置注入）/ 後置注入（A1）/ 不引入 reference（A2）。
- **數據來源**：FULL=R7（3-seed mean±std）、A1（單 seed）、A2（單 seed=42）。僅 FULL 列標 mean±std。
- 「不引入 reference」段是中心論點（reference 才是主貢獻），呼應第 1.2.1 節 → **維持**。
- 提醒：A2 的實作為「零張量取代 reference 特徵、保 adapter 參數量不變」，敘述可補一句「在不改變 Adapter 容量的前提下移除參考內容」以強化「資訊 vs 容量」的分離論證。

### 3.2 4.9.3 損失函數消融 — `tab:loss_ablation`【保留】
- 三列維持：完整模型 / 純 CE（C1）/ 取消 MFB（C2）。
- **數據來源**：FULL = R7（3-seed mean±std）、C1（單 seed）、C2（= R6 複用，mfb off, no-rcs，單 seed=42；R6 config 與「FULL 去掉 MFB」完全相同，**免費複用**）。
- rider/moto/bike per-class IoU 欄位由 eval 輸出直接填；ACDC 訓練集像素頻率（rider/moto/bike `[X.XXX]%`）由 `_ACDC_CLASS_FREQ` 導出填正文。
- 呼應第 4.5 節長尾失效 → 維持。

### 3.3 4.9.4 消融總結 — `tab:ablation_summary`【保留 + 吸收 §2】
- 7 列（R1–R7）；僅 FULL(R7) 標 mean±std（3-seed），其餘單 seed。
- 逐列趨勢段落吸收 §2 的 per-class 解碼與 LRH 論述。

### 3.4 4.9 節開頭段【微調】
- 原文為「每組消融僅切換單一模組」（leave-one-out）。現結構為**累積（R 系列）+ leave-one-out（4.9.1/4.9.3 的 A1/A2/C1/C2）並存**。開頭段補一句說明兩種消融並用：累積式觀察邊際貢獻（4.9.4），leave-one-out 在完整框架下隔離單一模組（4.9.1、4.9.3）。
- 補述 seed：僅完整框架（FULL=R7）以三組隨機種子重複，報平均與標準差；其餘各 run 單 seed=42。

---

## 4. 改寫後 4.9 節結構（定案）

```
4.9 消融實驗
├─ （開頭段，§3.4 微調：累積 + leave-one-out 並用）
├─ 4.9.1 注入機制（Adapter）消融   → tab:adapter_ablation        【保留】
├─ 4.9.2 損失函數消融              → tab:loss_ablation           【保留，原 4.9.3 改號】
└─ 4.9.3 消融總結                  → tab:ablation_summary        【保留，原 4.9.4 改號，吸收 §2】
```

> 原 4.9.2 解碼端消融刪除後，4.9.3→4.9.2、4.9.4→4.9.3 順序遞補（或保留原節號僅刪中間，依排版偏好）。

---

## 5. 改寫檢核清單

- [ ] 刪 4.9.2 小節 + `tab:decoder_ablation`（§1）
- [ ] grep `tab:decoder_ablation` 確認無孤兒 `\ref`
- [ ] 第 1.2.2 節呼應 → 搬到累積表 R3→R4 討論（§2）
- [ ] LRH（R4→R5）段誠實標註：僅報 mIoU 有限增益，未量測 boundary（§2）
- [ ] 4.9.1 adapter 表填值；僅 FULL 列標 mean±std；A2 補「不改變容量」一句（§3.1）
- [ ] 4.9.3 loss 表填值；C2 = R6 複用（mfb off, no-rcs, seed=42）；rider/moto/bike IoU + 像素頻率（§3.2）
- [ ] 4.9.4 累積表填值；僅 FULL(R7) 標 mean±std；7 列（R1–R7）；吸收解碼論述（§3.3）
- [ ] 開頭段補「累積 + leave-one-out 並用」與 seed 說明（§3.4）；長尾證據來自 loss 表 FULL vs C2（MFB 效益），不依賴 RCS
- [ ] 小節節號遞補（§4）
- [ ] 全節數值與 `aggregate_ablation.py` 三張表輸出一致
```
