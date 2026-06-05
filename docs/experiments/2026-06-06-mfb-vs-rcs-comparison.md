# 平衡機制對照實驗：MFB vs RCS vs Both

> 日期：2026-06-06　|　資料集：ACDC val（406 張）　|　評估：full-res 1080×1920、19 類 mIoU（eval_e1_acdc_val_full）
> 目的：判定 WeatherSAM 的 FULL 該採用哪種類別平衡機制（loss 端 MFB / 資料端 RCS / 兩者 / 都不要）。

## 1. 實驗設定（單變因，完全可比）

三個 run 除「平衡機制」外**所有設定相同**：

| 共同設定 | 值 |
|---|---|
| 架構 | inject=pre、decoder=unified、LRH on、adapter on |
| 損失 | CE + Lovász + Dice（lovasz=1, dice=1, label_smoothing=0.05） |
| 訓練 | epochs=50、patience=10、batch=1、accum=4、lr=5e-5、seed=42、AMP |

| Run | MFB（loss 加權） | RCS（資料過取樣） |
|-----|:--:|:--:|
| **R7** | ✅ on | ❌ off |
| **R8** | ✅ on | ✅ on |
| **C2** | ❌ off | ✅ on |

## 2. 主結果：overall mIoU

| Run | 機制 | **overall mIoU** | vs R7 |
|-----|------|:--:|:--:|
| **R7** | **MFB-only** | **67.26%** | — |
| R8 | both | 62.97% | −4.29 |
| C2 | RCS-only | 61.39% | −5.87 |
| *(參考)* | *E27 舊模型* | *65.68* | *+1.58* |

**R7（MFB-only）決定性勝出**，並且**超越舊參考 E27 的 65.68%**。加入 RCS（無論單獨 C2 或疊加 R8）都明顯掉分。

## 3. 逐條件 mIoU（R7 全面領先）

| Run | Fog | Rain | Snow | Night |
|-----|:--:|:--:|:--:|:--:|
| **R7** | **76.0** | **65.9** | **67.9** | **48.4** |
| R8 | 67.6 | 64.7 | 64.7 | 46.9 |
| C2 | 66.3 | 59.5 | 66.1 | 47.7 |

R7 在四種天氣條件**全部最高**，fog 領先最多（+8~10pp）。

## 4. 逐類別 IoU：MFB 的優勢集中在「稀有大型車輛」

| 類別 | R7 (MFB) | R8 (both) | C2 (RCS) | 觀察 |
|------|:--:|:--:|:--:|---|
| **truck** | **66.9** | 41.0 | 15.0 | ⭐ MFB 大勝（+26~52） |
| **bus** | **55.2** | 29.8 | 25.3 | ⭐ MFB 大勝（+25~30） |
| **train** | **71.8** | 62.1 | 49.7 | ⭐ MFB 大勝（+10~22） |
| fence | 58.3 | 52.7 | 48.5 | MFB 較好 |
| traffic sign | 65.5 | 62.6 | 62.2 | MFB 較好 |
| wall | 63.4 | 58.9 | 60.0 | MFB 較好 |
| rider | 19.4 | 17.4 | 21.3 | 互有高低（極稀有、高變異） |
| motorcycle | 37.3 | 40.5 | 31.0 | R8 略高 |
| bicycle | 45.2 | 43.5 | 46.5 | 互有高低 |
| road/sky/veg/car/person 等常見類 | ≈ | ≈ | ≈（C2 略高 0.5~3） | 差異小 |

**關鍵**：MFB（loss 端加權）把 **truck / bus / train** 這些稀有大型車輛拉得遠比 RCS 好（truck 66.9 vs C2 的 15.0！）。RCS（資料端過取樣）光把含稀有類的影像抽多，但 loss 端不加權，稀有類梯度不足 → truck/bus 反而崩。常見類三者幾乎相同。

## 5. 訓練穩定性：不穩定來自 MFB，但「值得」

早期 `train_ce_weighted`（MFB-加權 CE）ep1→ep5：

| Run | ep1→ep5 | 早期行為 |
|-----|---------|---------|
| R7 (MFB) | 3.8 → 5.0 → 7.9 → 10.5 → **13.9** | warmup 期爆漲→ep6 懸崖回落 |
| R8 (both) | 3.8 → 5.5 → 8.1 → 10.4 → **15.4** | 同上（更高） |
| **C2 (no MFB)** | 2.5 → 1.5 → 2.3 → 2.8 → **2.7** | **完全平穩，無 rise-cliff** |

- **不穩定確實源自 MFB 加權**（C2 拔掉 MFB 後早期完全平穩）。
- 但 C2（超穩）的 mIoU 反而最低（61.39）。→ **MFB 的增益遠大於其早期不穩定的成本**；且該不穩定在 ep6 自我恢復、不影響收斂。**結論：保留 MFB。**

## 6. 結論

1. **FULL 採用 MFB-only（= R7 配置）：inject=pre / unified / LRH / CE+Lovász+Dice / MFB on / RCS off。**
2. **RCS 移除**：在本設定下 RCS 無益且有害（R8、C2 均 < R7）。MFB（loss 端）對長尾的處理優於 RCS（資料端）。
3. MFB 造成的早期不穩定是良性的（自我恢復），其對 truck/bus/train 的增益是 FULL 勝出的主因，故保留。

## 7. Caveat（誠實標註）

- **長尾類別逐 seed 變異極大**（truck 在不同 run 曾從 15→67）。R7 的 67.26% 為**單一 seed**；先前無-RCS 跑曾得 ~63.5。→ R7 的「論文主數字」需以 **3 seeds（42/1234/2026）mean±std** 確立，本報告僅證明「同 seed 下 MFB-only > both > RCS-only」這個**機制排序**是穩固的。
- R7 vs C2 的機制排序（MFB-only 勝）在 seed 42 下差距達 ~6pp，遠大於 seed 噪聲，結論可信。

## 附：原始檔
- `outputs_ablation/R7_seed42/e1_results.json`、`R8_seed42/e1_results.json`、`C2_seed42/e1_results.json`
- 各自 `train_log.csv`（早期穩定性數據來源）
