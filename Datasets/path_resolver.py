"""CSV 路徑欄位的可攜化解析。

repo 內的 CSV 以佔位符記錄資料集位置，避免把開發機的絕對路徑寫死：

    ${DATASET_ROOT}/ACDC/rgb_anon/fog/train/...   # 外部資料集（ACDC / MUSES / Cityscapes ...）
    ${REPO_ROOT}/Datasets/features_train_all/...  # repo 內的特徵快取

使用前設定環境變數（未設定時採用預設值）：

    export DATASET_ROOT=/path/to/Datasets   # 預設 ~/Datasets
    export REPO_ROOT=/path/to/SAM_research  # 預設為本檔案的上上層目錄
"""
import os

# 需要展開的路徑欄位；CSV 依資料集不同只會出現其中幾個
PATH_COLUMNS = (
    'image_path',
    'ref_mask_path',
    'ref_image_path',
    'gt_path',
    'feature_path',
    'clear_feature_path',
    'invalid_mask',
)


def _defaults() -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'DATASET_ROOT': os.environ.get(
            'DATASET_ROOT', os.path.expanduser('~/Datasets')
        ),
        'REPO_ROOT': os.environ.get('REPO_ROOT', repo_root),
    }


def resolve_path(path):
    """展開單一路徑字串中的 ${DATASET_ROOT} / ${REPO_ROOT}。非字串原樣回傳。"""
    if not isinstance(path, str):
        return path
    for key, value in _defaults().items():
        path = path.replace('${%s}' % key, value)
    return path


def resolve_dataframe(df, columns=PATH_COLUMNS):
    """就地展開 DataFrame 中所有路徑欄位的佔位符，回傳同一個 df。"""
    mapping = _defaults()
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if series.dtype != object:
            continue
        for key, value in mapping.items():
            series = series.str.replace('${%s}' % key, value, regex=False)
        df[col] = series
    return df
