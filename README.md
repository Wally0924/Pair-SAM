# SAM_research
# WeatherSAM + Semantic Fusion Architecture & Training Flow

這份文件詳細記錄了 WeatherSAM 結合 SemanticFusionHead 的全景架構，特別針對在濃霧天氣場景下，如何利用 SAM 的 Zero-Shot 能力結合領域自適應微調 (Domain Adaptation) 來達成高精度的 19 類語意分割。

## 🗺️ 全景運作流程

### 階段一：資料與特徵萃取 (Data & Feature Extraction)
1. **輸入資料流 (DataLoader)** 
   每批次傳入：`image` (霧天原圖), `reference_mask` (晴天先驗遮罩/記憶), `location` (GPS 經緯度), 以及涵蓋 19 個類別的 `text_prompts`。
2. **視覺特徵編碼 (Frozen)**
   - 霧天原圖進入 `image_encoder` (ViT)，提取出豐富但模糊的圖像特徵。
   - `reference_mask` 進入 `mask_encoder`，萃取出幾何形狀的特徵。
   - *(此階段完美保留 SAM 原廠視覺能力，不參與訓練與權重更新)*

### 階段二：地理先驗與提示編碼 (Location & Prompt Encoding)
3. **地理特徵轉換 (Trainable Projection)**
   - GPS 座標送入 `location_encoder` (基於 GeoCLIP)。我們凍結其 Backbone 以維持其預訓練的世界觀，但**訓練它的 Projection 層**，將廣泛的地理知識翻譯為 SAM 能理解的神經訊號維度。
4. **文本提示轉換 (Trainable Projection)**
   - 19 個類別 (如 "road", "car") 送入 `text_encoder` (CLIP Text Encoder)。同樣凍結 Backbone，只**訓練 Projection 層**。
   - 隨後，文本與地理特徵會被串接 (Concat) 成為強大的混合 Prompt (Sparse Embeddings)。

### 階段三：跨視角特徵融合 (Cross-View Fusion)
5. **天氣狀態動態對齊 (Trainable)**
   - `fusion_module` (CrossViewAlignment) 負責將「晴天幾何特徵」對齊「霧天圖像特徵」，在迷霧中找到對應的參考座標。
   - `gate_module` (GatedFusion) 類似於注意力閘門，根據當下霧的濃淡，動態決定「要拿多少比例的晴天記憶來填補現在的視覺盲區」。
   - *(這是模型適應惡劣天氣的最核心訓練部位)*

### 階段四：原生遮罩解碼 (Mask Decoding)
6. **獨立遮罩生成 (Frozen)**
   - 混合了 Prompt 與融合後的高階圖像特徵，一起送入 SAM 的心臟 `mask_decoder`。
   - SAM 針對 19 個類別**獨立**吐出 19 張 256x256 的 `low_res_logits`。
   - *(我們凍結這裡，確保 SAM 不會忘了如何畫出銳利的高頻物件邊緣)*

### 階段五：高解析度還原與語意縫合 (Semantic Fusion & Prediction)
7. **高解析度還原**
   - 呼叫原生 `postprocess_masks`，將 256x256 的分數精準雙線性放大並裁切回 1024x1024 原始解析度。
8. **Semantic Fusion Head 空間打架消除 (Trainable)**
   - 將最佳的 19 張預測圖堆疊成 `(Batch, 19, 1024, 1024)` 的立體張量，送入全新的 `SemanticFusionHead`（由 1x1 和 3x3 卷積組成）。
   - Head 會觀察所有類別的空間分佈，學習消除邊界衝突（例如重疊的「車」與「路」）。最終輸出一張**平滑且互斥**的全景潛在特徵圖 (Logits)。

### 階段六：損失計算與梯度回傳 (Loss & Optimization)
9.  **全區域交叉熵優化 (Semantic CrossEntropyLoss)**
    - 拿經過 Head 處理後的 `(19, 1024, 1024)` Logits，直接與真實的 `(1024, 1024)` Ground Truth 計算 `nn.CrossEntropyLoss`。
    - 設定 `ignore_index=255`，使得那些無法判定或不屬於 19 類的 Void 區域被完美忽略，不參與 Loss 計算。
10. **精準的反向傳播 (Parameter-Efficient Fine-Tuning)**
    - 算出梯度一路往回傳，但被我們設下的 Freeze 護城河擋下來。
    - 最終，整個龐大的 `WeatherSAM` 中，只有最關鍵的 **5 個部位**吸收經驗並更新權重：
      - `semantic_fusion_head`
      - `gate_module`
      - `fusion_module`
      - `location_encoder.output_projection`
      - `text_encoder.projection`

---
*Created during architecture restructuring for Semantic Cross Entropy & Head-Tuning.*
