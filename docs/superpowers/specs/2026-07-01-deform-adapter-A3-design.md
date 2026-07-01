# WeatherSAM 雙向可變形 Adapter(A3)設計

- 日期:2026-07-01
- 分支:`feat/adapter-redesign`
- 狀態:設計已與使用者確認,待轉 writing-plans

## 1. 背景與動機

WeatherSAM 目前的 encoder 端 adapter(`MultiScaleCrossAttnInjector`)是 ViT-Adapter
的**單向簡化版**:只有 Injector、K/V 被 `adaptive_avg_pool2d` 塌成單一 32×32 grid、
用 vanilla `nn.MultiheadAttention`。對照 ViT-Adapter(ICLR 2023)原始實作後確認缺兩件事:

1. **Extractor(反向抽取路徑)**——ViT-Adapter 精度主力(原論文 Table 6:Extractor 單項 +2.1 AP)。
2. **多尺度可變形互動**——原版 K/V 是 1/8+1/16+1/32 三尺度 + MSDeformAttn 跨尺度取樣;
   我們把尺度塌掉、且用全域 vanilla attention。

本設計把 adapter 重建為**忠於 ViT-Adapter 的雙向多尺度可變形 adapter**,但
Spatial Prior Module(SPM)替換為既有的 **UAWarpC 跨天氣參考對齊**(CMAAlignment.pre_align)。

## 2. 目標與硬約束

**目標**:encoder 端注入器補齊「多尺度 + 雙向」兩大缺口,達到與 ViT-Adapter 對等的
adapter 機制,同時保留 WeatherSAM 的 per-class 動態核 decoder 創新。

**硬約束(不可違反)**:
- **非侵入**:不改 ViT 任何一層權重/結構,只透過 forward hook 加殘差。
- **單尺度輸出(方案 A)**:decoder / LRH(ResidualDWConvFusion)/ 輸出路徑一律不動。
  Extractor 精修的參考分支 `c` **不進輸出**,僅跨 stage 強化注入。
- **無回歸**:`use_vgg_adapter=False`、`use_reference=False` 兩個 ablation 開關仍可用。
- **4090 24GB 可訓**:bf16 + gradient checkpoint 下不 OOM。

### 為何是方案 A(而非讓 c 進輸出)
WeatherSAM 的 decoder 是 Mask2Former-style **per-class 動態核**(單一 1/16 尺度輸入),
與 ViT-Adapter 的「多尺度金字塔 → dense head」是兩套不相容的輸出哲學。讓 `c` 進輸出會
逼迫更換 decoder,犧牲 WeatherSAM 的核心創新。方案 A 只在 encoder 端補雙向機制,
輸出仍走 ViT → SAM neck → decoder → LRH,守住創新與非侵入。

## 3. 整體架構

```
forward():
  pre_align(img_curr, img_ref) → {l2(1/8,256ch), l3(1/16,512ch), mask(信心)}  [no_grad,沿用]
  adapter.set_features({l2,l3,mask}, H, W):
       RPM 建 c₀(3 尺度 token@1280 + level_embed + per-token 信心)
       計算 deform_inputs(reference points / spatial_shapes / level_start_index)
       重置 self._c=c₀, self._stage=0
  image_encoder(x):
       pre-hook @block0 : x = Injector₀(x, c₀)
       blocks 0–7 run
       post-hook@block7 : c₁ = Extractor₀(c₀, x) ; self._c=c₁
       pre-hook @block8 : x = Injector₁(x, c₁)   … block15 post → c₂
       pre-hook @block16: x = Injector₂(x, c₂)   … block23 post → c₃
       pre-hook @block24: x = Injector₃(x, c₃)    ← 末端只注入,不 extract
       → neck → image_embeddings(輸出路徑不變)
  decoder / LRH 不變
```

- ViT-H 32 層切成 4 組(每組 8 層):`[0-7][8-15][16-23][24-31]`。
- 注入在**組前**(pre-hook @0/8/16/24),抽取在**組後**(post-hook @7/15/23)。
- **4 Injector + 3 Extractor**:末組(24-31)只注入,因 c₄ 無消費者(方案 A)。
- SAM 的 global attention block(7/15/23/31)恰落各組末端 → 注入的參考在 extract 前
  被該組 global block 全域傳播一次。

## 4. 新模組(新檔 `segment_anything/modeling/deform_adapter.py`)

### 4.1 ReferencePriorModule(RPM)— 取代 SPM
```
輸入: l2(B,256,128,128), l3(B,512,64,64), mask(B,1,H,W)  ← pre_align 已乘過 hard_mask
c2(1/8) : l2 → 1×1 conv → 1280            (128×128 = 16384 tok)
c3(1/16): l3 → 1×1 conv → 1280            ( 64×64  = 4096  tok)
c4(1/32): l3 → stride-2 conv → 1280       ( 32×32  = 1024  tok)   ← 決策②:降採 l3
c = flatten+concat([c2,c3,c4]) + level_embed(3,1280)
conf_per_token = pool(mask) 對齊三尺度,供 injector value 加權(決策③)
```

### 4.2 Injector(Q=ViT, K/V=c, n_levels=3)
```
attn = MSDeformAttn(query_norm(x.detach()), ref_pts, feat_norm(c), shapes_3lvl)
                     ↑ query 用 detach:只決定取樣位置/權重,不讓 adapter 梯度重塑 ViT
x = x + softplus(gate)·attn           ← 殘差 x 有梯度(ViT 仍由主任務 loss 訓練)
```
- **gate**:沿用 `softplus(init≈0.05)` + trainer gate warmup(**不用** ViT-Adapter 的
  zero-init gamma;既有實作已驗證 zero-init 在 softplus 下梯度死鎖,且 trainer 已支援 warmup)。
- `deform_ratio=0.5`(value 投影減半省記憶體)。

### 4.3 Extractor(Q=c, K/V=ViT, n_levels=1)+ ConvFFN
```
attn = MSDeformAttn(query_norm(c), ref_pts, feat_norm(x.detach()), shapes_1lvl)
                     ↑ ViT feat 用 detach:把 ViT 當固定語境讀,不回灌梯度到 ViT
c = c + attn
c = c + drop_path(ConvFFN(ffn_norm(c)))   ← ConvFFN 內逐尺度 DWConv(把 c 拆回 3 個 2D grid)
```
- 末組不建 extractor;`extra_extractors` 不使用(為輸出金字塔而生,方案 A 不需要)。

### 4.4 梯度/保護策略(關鍵取捨,已確認保留)
| 路徑 | ViT 梯度 | 說明 |
|---|---|---|
| Injector attention(offset/weight) | ✗ detach | adapter 不透過注意力重塑 ViT |
| Injector 殘差 x | ✓ | ViT 仍由主 loss 訓練,不受 adapter 影響 |
| Extractor feat=ViT | ✗ detach | 把 ViT 當固定語境,只更新 c |
| Extractor c / 所有 adapter 參數 | ✓ | adapter 正常學習 |

**核心保證**:adapter 兩方向都不用梯度改 ViT;ViT 是否 fine-tune 由 trainer 從 decoder 側
主 loss 決定,與 adapter 解耦。忠於 ViT-Adapter 的 non-invasive 精神,亦守住「凍結 SAM encoder」立場。

## 5. Hook 生命週期與狀態機
- `set_features()`:建 c₀、算 deform_inputs、重置狀態(每 forward 一次)。
- pre-hook @0/8/16/24:注入,回傳改動後 x。
- post-hook @7/15/23:抽取,更新 `self._c`,**回傳原 output 不變**(extractor 只讀 ViT、不改 ViT 輸出)。
- SAM block I/O 為 `(B,H,W,C)`,沿用既有 reshape 慣例。
- 沿用既有診斷欄位:`_last_inject_cos_sim` / `_last_gate_val` / `_last_delta_norm_ratio` / per-stage。

## 6. 已定案決策
| # | 問題 | 決定 |
|---|---|---|
| ① | MSDeformAttn CUDA vs pytorch fallback | **先驗證 CUDA 編譯(里程碑 0);失敗即用 pytorch fallback**。兩者 vendor 進 `ops/`,runtime 自動選 |
| ② | 第三尺度(1/32)來源 | **RPM 內對 l3 做 stride-2 conv 降採**(不動 CMA backbone) |
| ③ | confidence 接進 deformable | **value 特徵投影前乘 per-token 信心**(低信心參考注入弱、可微)。extractor 端不加信心 |

## 7. 記憶體(最大風險)與緩解階梯
c 的 1/8 尺度 16384 tok@1280 為主壓力(ViT-Adapter 在 A100 40GB 跑全解析度;此為 4090 24GB + ViT-H,更緊)。
**里程碑 0 先 dry-run 量測**,若 OOM 依序啟用:
1. interaction 開 `with_cp`(grad checkpoint injector/extractor)。
2. `deform_ratio=0.5`(已預設)。
3. 1/8 尺度預 pool 到 96×96 或 64×64(記錄取捨,誠實揭露)。
4. 退回 2 尺度(1/16+1/32),放棄 1/8。

## 8. 要動的檔案(外科式)
- **新增** `segment_anything/modeling/deform_adapter.py`:RPM / Injector / Extractor / ConvFFN / 狀態機。
- **新增** `ops/`(從 ViT-Adapter vendored 的 MSDeformAttn,含 pytorch fallback)。
- **改** `segment_anything/modeling/weather_sam.py`:建構新 adapter、hook 註冊改為
  pre@0/8/16/24 + post@7/15/23、`set_features` 簽名、deform_inputs 接線。
- **改** `segment_anything/build_weather_sam.py`:建構新 adapter + config。
- **沿用** `segment_anything/modeling/fusion.py::pre_align`(已輸出 l2/l3/mask,無需改)。

## 9. Ablation 相容
- `use_vgg_adapter=False` → 不註冊 hook(機制不變)。
- `use_reference=False` → RPM 輸出零化 c(保留結構/參數量)。
- ⚠️ `SameImageAdapterInjector` 基線需同步加 extractor 鏡像才能與新 FULL 對照,列為 follow-up。

## 10. 驗證計畫(里程碑)
- **里程碑 0(去風險,必先過)**:編譯/匯入 MSDeformAttn(CUDA 或 fallback)+ dummy forward 量記憶體。
- 單元:RPM 三尺度/level_embed/信心形狀;Injector 對 ViT 流 shape-preserving;
  Extractor 更新 c shape-preserving;deform_inputs 正確性。
- 整合:全 forward 無 NaN、輸出形狀與舊版一致;梯度檢查(adapter 有梯度、ViT 依 detach 設計受保護)。
- 記憶體:4090 / 1024² / bf16 / grad ckpt dry-run 不 OOM。
- 數值:gate warmup 下初期注入接近零(gate≈0.05),診斷欄位正常。

## 11. 範圍聲明
大改(新 CUDA 依賴、新模組、hook 重構),但 decoder/LRH/輸出全不動,仍是單一連貫實作計畫。
WeatherSAM 的 per-class 動態核 decoder 創新完整保留。
