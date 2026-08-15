"""確保 repo 內的資料索引不含開發機的絕對路徑。

公開 repo 的 CSV / JSON 一律以 ${DATASET_ROOT}、${REPO_ROOT} 佔位符記錄資料集位置，
由 Datasets/path_resolver.py 在讀取時展開。這組測試防止有人不慎把絕對路徑寫回索引檔。
"""
import json
import os
import re
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASETS = os.path.join(_REPO_ROOT, 'Datasets')
sys.path.insert(0, _DATASETS)

from path_resolver import PATH_COLUMNS, resolve_dataframe, resolve_path  # noqa: E402

# 使用者家目錄形式的絕對路徑，例如 /home/alice/... 或 /Users/alice/...
_ABSOLUTE_HOME = re.compile(r'/(?:home|Users)/[^/\s,"]+/')


def _tracked_csvs():
    """repo 內的資料索引 CSV（含 foggy_0.02 等子目錄）。"""
    found = []
    for dirpath, _, filenames in os.walk(_DATASETS):
        # 特徵快取目錄不含索引檔，且體積龐大
        if 'features' in os.path.basename(dirpath) or '__pycache__' in dirpath:
            continue
        for name in filenames:
            if name.endswith('.csv'):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


@pytest.mark.parametrize('csv_path', _tracked_csvs(), ids=os.path.basename)
def test_csv_has_no_absolute_home_path(csv_path):
    """CSV 的路徑欄位不得含開發機絕對路徑。"""
    df = pd.read_csv(csv_path)
    for col in PATH_COLUMNS:
        if col not in df.columns or df[col].dtype != object:
            continue
        offenders = df[col].dropna().astype(str)
        offenders = offenders[offenders.str.contains(_ABSOLUTE_HOME, regex=True)]
        assert offenders.empty, (
            f'{os.path.relpath(csv_path, _REPO_ROOT)} 欄位 {col} 有 {len(offenders)} 筆絕對路徑，'
            f'首筆：{offenders.iloc[0]}'
        )


def test_class_presence_keys_are_portable():
    """class_presence.json 的 key 需與 CSV 的 gt_path 同樣使用佔位符。"""
    path = os.path.join(_DATASETS, 'class_presence.json')
    if not os.path.isfile(path):
        pytest.skip('class_presence.json 不存在')
    with open(path) as f:
        data = json.load(f)
    offenders = [k for k in data['presence'] if _ABSOLUTE_HOME.search(k)]
    assert not offenders, f'{len(offenders)} 個 key 含絕對路徑，首個：{offenders[0]}'


def test_class_presence_keys_match_train_csv():
    """展開後的 presence key 必須覆蓋訓練 CSV 的每一筆 gt_path，否則 --rcs 會 KeyError。"""
    cp_path = os.path.join(_DATASETS, 'class_presence.json')
    csv_path = os.path.join(_DATASETS, 'acdc_adverse_ref_rgb_train.csv')
    if not (os.path.isfile(cp_path) and os.path.isfile(csv_path)):
        pytest.skip('class_presence.json 或訓練 CSV 不存在')
    with open(cp_path) as f:
        presence = {resolve_path(k) for k in json.load(f)['presence']}
    gt_paths = resolve_dataframe(pd.read_csv(csv_path))['gt_path'].tolist()
    missing = [g for g in gt_paths if g not in presence]
    assert not missing, f'{len(missing)} 筆 gt_path 缺少 presence，首筆：{missing[0]}'


def test_resolver_expands_both_placeholders(monkeypatch):
    monkeypatch.setenv('DATASET_ROOT', '/data/sets')
    monkeypatch.setenv('REPO_ROOT', '/srv/repo')
    assert resolve_path('${DATASET_ROOT}/ACDC/a.png') == '/data/sets/ACDC/a.png'
    assert resolve_path('${REPO_ROOT}/Datasets/f.pt') == '/srv/repo/Datasets/f.pt'


def test_resolver_passes_through_non_string():
    """NaN 等非字串值需原樣回傳，避免 dropna 前的欄位被破壞。"""
    assert resolve_path(float('nan')) != ''
    assert resolve_path(None) is None
