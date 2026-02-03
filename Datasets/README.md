# 📜 Scripts Description

### 1. `sperate_data.py`

**功能：** 資料集分割 (Dataset Partitioning)
- 負責將原始的資料集依照預設比例（例如 80% 訓練、10% 驗證、10% 測試）進行隨機劃分。
- 確保訓練集、驗證集與測試集之間無資料重疊，並輸出分割後的檔案清單或移動檔案至對應目錄，建立標準化的資料結構。

### 2. `generate_csv.py`

**功能：** 資料索引生成 (Index Generation)
- 掃描指定的資料夾結構，自動配對 原始影像 (Input Image)、真值遮罩 (Ground Truth) 與 參考遮罩 (Reference Mask)。
- 將配對好的檔案路徑彙整並寫入 CSV 檔案。此 CSV 檔將作為 PyTorch DataLoader 的主要輸入來源，取代傳統的資料夾遍歷方式，提升讀取效率。

### 3. `check_data_integrity.py`

**功能：** 資料完整性驗證 (Integrity Validation)
- 讀取生成的 CSV 索引檔，逐筆檢查所有記錄的檔案路徑是否真實存在。
- 嘗試讀取影像以檢測檔案是否損毀 (Corrupted)，並驗證影像與遮罩的尺寸是否一致，預防在模型訓練過程中因 I/O 錯誤導致中斷。


### 4. `add_gps_to_csv.py`

**功能：** 地理資訊整合 (GPS Metadata Integration)
- 解析 CSV 中的影像檔名（基於 Cityscapes 命名規則），自動關聯對應的 `vehicle_sequence.json` 元數據檔案。
- 提取每張影像拍攝時的 GPS 經緯度 (Latitude, Longitude)。
- 將地理座標作為新欄位追加至 CSV 中，使模型能利用地理位置資訊進行檢索增強 (Retrieval-Augmented) 或位置編碼訓練。