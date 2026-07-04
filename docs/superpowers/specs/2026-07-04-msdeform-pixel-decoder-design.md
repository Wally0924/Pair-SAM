# WeatherSAM 加入 MSDeformAttn Pixel Decoder — Design Spec

> **狀態:** 待使用者審核 → 通過後進 `superpowers:writing-plans`。
> **前置 plan:** [`2026-07-03-m2f-decoder-redesign.md`](../plans/2026-07-03-m2f-decoder-redesign.md)。本 spec **推翻**其中「不移植 MSDeformAttn pixel decoder（使用者定案：FPN 直入 decoder）」的決定。

## 背景與動機

現行 `decoder_arch='m2f'` 路徑是 **SimpleFPN 直入 masked-attention decoder**，中間無 pixel decoder——刻意省掉 Mask2Former 官方的跨尺度融合層。經討論確認：

- 幾乎所有使用 masked-attention decoder 的工作都保留一個會**跨尺度融合**的 pixel decoder（Mask2Former 官方為 MSDeformAttn；ViTDet / ViT-Adapter 在 backbone 端造金字塔後仍餵給它）。
- 現行 SimpleFPN 產出的 `[f32,f16,f8]` 是**同源重採樣的偽多尺度**，缺乏跨尺度資訊交互。

**目標：** 忠實移植 Mask2Former 官方 `MSDeformAttnPixelDecoder`，插入 SimpleFPN 與 M2FDecoder 之間，做跨尺度融合並產生 1/4 mask features。

## 決策紀錄（使用者定案）

1. **整合方式：直接取代。** 移除 FPN 直入路徑，`m2f` 路徑一律經過 pixel decoder。**不**保留 `use_pixel_decoder` 建構旗標。
2. **版本：官方 MSDeformAttnPixelDecoder**（6 層 deformable encoder + FPN lateral），非輕量 FPN 版。
3. **顯存退路：** memcheck 破 20 GB 時，可將 deformable encoder 深度由官方 6 層降至約 4 成（≈ 3–4 層），仍屬官方可配置範圍。

## 目標架構與資料流

```
ViT → image_embeddings (1,256,64,64)
  → SimpleFPN                    → feats=[f32,f16,f8], f4          （不變）
  → MSDeformAttnPixelDecoder     → feats_融合=[f32,f16,f8], mask_features(1/4)   ★新增
  → M2FDecoder(feats_融合, mask_features, text_feat, cond_tok)      （不變）
  → sem_lr = softmax(class)⊗sigmoid(mask) → postprocess_masks       （不變）
```

**外科級插入的依據——SimpleFPN 與 M2FDecoder 皆不需修改：**

- SimpleFPN 現有輸出 `[f32,f16,f8], f4` 剛好對齊官方 pixel decoder 的 4 尺度介面：前三尺度（stride 8/16/32）當 deformable encoder 輸入，f4（stride 4）當 FPN lateral 來源產 mask_features。
- 官方 pixel decoder 輸出 `multi_scale_features`（coarse→fine 排序 `[stride32,16,8]`）+ `mask_features`（stride 4），**形狀完全對齊** `M2FDecoder.forward(feats, mask_features, ...)` 現有簽名。

## 元件設計

### 新模組：`modeling/msdeform_pixel_decoder.py`（vendor）

| 項目 | 內容 |
| --- | --- |
| 上游出處 | Mask2Former `mask2former/modeling/pixel_decoder/msdeformattn.py`：`MSDeformAttnPixelDecoder`、`MSDeformAttnTransformerEncoderOnly`、`MSDeformAttnTransformerEncoderLayer` |
| 論文 | Cheng et al., Mask2Former, CVPR 2022；deformable attn 源自 Zhu et al., Deformable DETR, ICLR 2021 |
| 授權 | MIT（Mask2Former repo）；保留上游 copyright 行 |
| 複用 | `ops/ms_deform_attn.py`（既有純 PyTorch `MSDeformAttn`）、`m2f_decoder.py` 之 `PositionEmbeddingSine` |

**移植子元件：**

- `MSDeformAttnTransformerEncoderLayer`：`self_attn = MSDeformAttn` + FFN（逐行同上游）。
- `MSDeformAttnTransformerEncoderOnly`：`level_embed` 參數、`get_reference_points`、6 層 encoder 堆疊。
- `MSDeformAttnPixelDecoder` 主體：
  - `input_proj`：3 個 `1×1 Conv + GroupNorm(32)`，投影三尺度到 conv_dim=256。
  - encoder 前向：flatten 三尺度 → 加 `level_embed` + 正弦 PE → 6 層 deformable encoder → split 回三尺度 `[f32',f16',f8']`。
  - FPN lateral：從 f8'（stride 8）top-down 加 f4（stride 4）lateral → `mask_features`（stride 4，256-d）。
  - 回傳 `([f32',f16',f8'], mask_features)`。

**[WeatherSAM adaptation] 清單（每項檔頭列出 + 行內註記）：**

1. 移除 detectron2 依賴（`@configurable`、`ShapeSpec`、`Conv2d`/`get_norm` wrapper、`Registry`）→ 純 `nn.Module` + `nn.Conv2d` + `nn.GroupNorm`。
2. 輸入介面由「backbone features dict」改為「`(feats:list[3], f4:Tensor)`」——直接吃 SimpleFPN 輸出，省去 dict 排序。
3. `num_feature_levels=3` 固定；`common_stride=4`；輸出改回 `(list, tensor)` tuple 對齊 M2FDecoder。
4. deformable attn 複用本 repo `ops.MSDeformAttn`（純 PyTorch），不引入上游 CUDA op。

### 三處外科修改

| 檔案 | 位置 | 修改 |
| --- | --- | --- |
| `modeling/weather_sam.py` | `__init__`（約 38–48 行） | 新增 `pixel_decoder: nn.Module = None` 參數 + `self.pixel_decoder =`；m2f 分支 assert 併入 pixel_decoder 非 None |
| `modeling/weather_sam.py` | forward m2f 分支（245–246 行間） | 插入 `feats, mask_features = self.pixel_decoder(feats, mask_features)` |
| `modeling/build_weather_sam.py` | 128–141 行 | 建 `MSDeformAttnPixelDecoder(...)`，傳入 `WeatherSAM(pixel_decoder=...)` |
| `modeling/__init__.py` | export | 加 `MSDeformAttnPixelDecoder` |

**不改動：** `simple_fpn.py`、`m2f_decoder.py`、encoder 端所有檔案、legacy decoder 路徑。

## 錯誤處理與邊界

- pixel decoder 為 m2f 路徑**必要元件**：build 未提供時，m2f 分支 assert 明確報錯（比照現有 simple_fpn/m2f_decoder assert 風格）。
- reference points / valid ratio：B=1、無 padding，`input_padding_mask=None`，`valid_ratios=1`（沿用上游 `get_valid_ratio` 但單圖恆為 1）。

## 驗證計畫（成功標準）

1. **形狀單元測試** `tests/test_pixel_decoder.py`（TDD 先寫）：
   - 輸入 `([f32(1,256,32,32), f16(1,256,64,64), f8(1,256,128,128)], f4(1,256,256,256))`。
   - 斷言輸出三尺度 shape 與輸入三尺度逐一相同；`mask_features` 為 `(1,256,256,256)`。
   - 斷言可反向傳播（`loss.backward()` 後三尺度 input 有 grad）。
2. **整合測試** `tests/test_m2f_forward.py`（更新）：完整 forward，斷言 `pred_logits (1,19,20)`、`pred_masks (1,19,256,256)`、`sem_lr (1,19,256,256)` 形狀不變（pixel decoder 只改內容不改對外形狀）。
3. **memcheck** `scripts/memcheck_m2f.py`：峰值 **≤ 20 GB**。破表時套用退路（encoder 降至 3–4 層）再驗。
4. **vendor diff 對照**：下載上游原檔至 scratch，逐段核對 vendored 主體，僅標註 adaptation 的行可不同。

## 全域約束（沿用前置 plan）

- 一律 `conda run -n sam_env python/pytest ...`。
- Encoder 端一行不改；legacy decoder 路徑維持可運行。
- 純 PyTorch，不新增外部依賴（複用既有 `ops.MSDeformAttn`）。
- 影像 1024×1024、19 類、AMP + GradScaler(2048)、B=1 + 梯度累積、4090 ≤ 20 GB。
- 外科手術式修改，不順手重構。

## 檔案異動總表

| 檔案 | 動作 |
| --- | --- |
| `modeling/msdeform_pixel_decoder.py` | 新增（vendor） |
| `modeling/weather_sam.py` | 改（__init__ + forward 各一處） |
| `modeling/build_weather_sam.py` | 改（組裝） |
| `modeling/__init__.py` | 改（export） |
| `tests/test_pixel_decoder.py` | 新增 |
| `tests/test_m2f_forward.py` | 改（整合斷言） |
| `scripts/memcheck_m2f.py` | 重跑（不改或微調層數） |
