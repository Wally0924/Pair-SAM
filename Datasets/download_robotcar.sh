#!/usr/bin/env bash
# =============================================================================
# download_robotcar.sh
# -----------------------------------------------------------------------------
# 下載 RobotCar Correspondence（跨季節對應）資料集，對齊 Refign / CMA 的實驗設定。
#
# 官方將資料拆成兩處：
#   1) 標註 / 對應資料  → CIIRC (CTU Prague) 伺服器，用 wget 遞迴抓
#      correspondence_data/ + segmented_images/ + LICENSE.txt + README.txt
#   2) 原始影像 images/  → Google Drive，用 gdown 抓
#
# 最終目錄結構（符合 brdav/cma、brdav/refign 的 README）：
#   $DATA_DIR/RobotCar
#   ├── images/               # dawn, dusk, night, night-rain, ...
#   ├── correspondence_data/
#   ├── segmented_images/     # training / validation / testing
#   ├── LICENSE.txt
#   └── README.txt
#
# 用法：
#   ./download_robotcar.sh [DATA_DIR]
#   # DATA_DIR 省略時預設為 ./data
#
# 授權：CC BY-NC-SA 4.0，僅限非商業學術用途。
# =============================================================================
set -euo pipefail

# ---- 參數 -------------------------------------------------------------------
DATA_DIR="${1:-./data}"
RC_DIR="${DATA_DIR}/RobotCar"

CIIRC_URL="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Cross-Seasons-Correspondence/ROBOTCAR/"
GDRIVE_IMAGES_FOLDER="https://drive.google.com/drive/folders/19yUB49EliCnWThuN2HUukIryX47JWmQp"

echo "=========================================================="
echo " RobotCar Correspondence 下載"
echo "   目標目錄：${RC_DIR}"
echo "=========================================================="
mkdir -p "${RC_DIR}"

# ---- 步驟 1：CIIRC 伺服器（wget 遞迴） --------------------------------------
# --cut-dirs=5 去掉路徑前綴 public/projects/2020VisualLocalization/
#              Cross-Seasons-Correspondence/ROBOTCAR，讓檔案直接落在 RC_DIR 下。
echo ""
echo "[1/2] 從 CIIRC 抓 correspondence_data / segmented_images / 授權檔 ..."
wget \
  --recursive --no-parent --no-host-directories --cut-dirs=5 \
  --reject "index.html*" \
  --continue \
  --directory-prefix "${RC_DIR}" \
  "${CIIRC_URL}"
echo "    ✓ CIIRC 部分完成"

# ---- 步驟 2：Google Drive 影像（gdown） ------------------------------------
echo ""
echo "[2/2] 從 Google Drive 抓 images/ ..."

# 確認 gdown 可用；base 沒有的話嘗試 refign / cma conda env，再退回 pip 安裝。
if ! command -v gdown >/dev/null 2>&1; then
  echo "    未偵測到 gdown，嘗試尋找可用環境 ..."
  GDOWN_BIN=""
  for env in refign cma; do
    cand="${HOME}/miniconda3/envs/${env}/bin/gdown"
    if [ -x "${cand}" ]; then
      GDOWN_BIN="${cand}"
      echo "    → 使用 conda env '${env}' 的 gdown"
      break
    fi
  done
  if [ -z "${GDOWN_BIN}" ]; then
    echo "    → 找不到現成 gdown，改用 pip 安裝到目前 Python 環境 ..."
    python -m pip install --quiet --upgrade gdown
    GDOWN_BIN="$(command -v gdown)"
  fi
else
  GDOWN_BIN="$(command -v gdown)"
fi

# 下載整個資料夾到 RobotCar/images。
#   --remaining-ok：資料夾內檔案 >50 個時不中斷。
#   --continue    ：支援續傳。
"${GDOWN_BIN}" --folder "${GDRIVE_IMAGES_FOLDER}" \
  --output "${RC_DIR}/images" \
  --remaining-ok --continue

echo "    ✓ Google Drive 部分完成"

# ---- 完成提示 ---------------------------------------------------------------
echo ""
echo "=========================================================="
echo " 完成。請檢查目錄結構："
echo "   ${RC_DIR}/{images,correspondence_data,segmented_images}"
echo ""
echo " 注意事項："
echo "   * Google Drive 大資料夾偶爾會回報配額超限（quota exceeded），"
echo "     若中途失敗，等數小時或改天重跑本腳本即可（已開 --continue 續傳）。"
echo "   * gdown 可能把影像放進多一層子資料夾，請對照上面的結構自行確認/搬移。"
echo "   * 資料僅限非商業學術用途（CC BY-NC-SA 4.0）。"
echo "=========================================================="
