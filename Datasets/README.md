# 資料索引與前處理

## 資料集取得

本 repo 只提供資料索引 CSV，不散布資料集影像。請自行向各資料集官方申請並下載：

| 資料集 | 用途 | 取得方式 |
|--------|------|----------|
| [ACDC](https://acdc.vision.ee.ethz.ch/) | 主要訓練與評估（fog / rain / snow / night） | 官網註冊後下載 |
| [MUSES](https://muses.vision.ee.ethz.ch/) | 跨資料集泛化評估（weather × time-of-day 八種條件） | 官網註冊後下載 |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 晴天預訓練與 GT | 官網註冊後下載 |
| [Foggy Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) | 合成霧氣訓練 | 隨 Cityscapes 提供 |
| [Dark Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) | 夜間評估 | 官網下載 |

下載後請維持各資料集的原始目錄結構，統一放在同一個根目錄下：

```text
$DATASET_ROOT/
├── ACDC/
├── MUSES/
├── Cityscapes/
├── Cityscapes_foggy/
└── Dark_Zurich/
```

## 路徑設定

CSV 中的路徑以佔位符記錄，執行前需設定環境變數：

```bash
export DATASET_ROOT=/path/to/your/Datasets   # 預設 ~/Datasets
export REPO_ROOT=/path/to/SAM_research       # 預設為本 repo 根目錄，通常不需設定
```

兩個佔位符的意義：

- `${DATASET_ROOT}` — 外部資料集的原始影像與標註
- `${REPO_ROOT}` — repo 內的預計算特徵快取（`Datasets/features_*`，需自行以 `precompute_features.py` 產生）

路徑展開由 `path_resolver.py` 處理，`pair_dataloader.py` 與 `precompute_features.py` 讀取 CSV 後會自動套用，一般情況下無需手動呼叫。若自行撰寫分析腳本讀取這些 CSV：

```python
import pandas as pd
from path_resolver import resolve_dataframe

df = resolve_dataframe(pd.read_csv('Datasets/acdc_adverse_ref_rgb_val.csv'))
```

---

## 📜 Scripts Description

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