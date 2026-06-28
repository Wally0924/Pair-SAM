"""把既有的 Dark Zurich test 單模態預測重新打包成 CodaLab 23553 規範格式。

背景
----
Dark Zurich test 的唯一官方評測伺服器是 CodaLab competition 23553（GCMA/UIoU）。
它要求提交的 .zip **頂層含三個子目錄**，缺一即 evaluation 失敗：

  labelTrainIds/          ← 語意標籤（Cityscapes trainIds 0..18），uint8 PNG
  confidence/             ← 置信度圖，uint16 PNG（0..65535 線性對應 0.0..1.0）
  labelTrainIds_invalid/  ← 含 invalid（值 255）的語意標籤，uint8 PNG

三者分別算出 IoU / Average UIoU / UIoU 三個指標。本專案目標僅取**標準 IoU**
（由 labelTrainIds 計算），因此採「最小正確解」：confidence 與 invalid 以不影響
IoU 的方式填充，使提交合法且 IoU 完全正確。

  * labelTrainIds        ：直接沿用既有預測（已合規，0..18、1080x1920、uint8）。
  * labelTrainIds_invalid：複製 labelTrainIds，**不標任何 255**。依規範「無 invalid
                           像素時 UIoU 定義上等於 IoU」，故不影響 IoU。
  * confidence           ：常數 65535（即信心 1.0）。只影響 Average UIoU，不碰 IoU。

佈局：三個子目錄內均**扁平**放置 151 個檔案（不巢狀 test/night/{seq}/）。
規範以 glob ``{sequence}_frame_{frame:0>6}*.png`` 在子目錄中尋找唯一匹配；扁平佈局
在「遞迴走訪」與「單層列舉」兩種 eval 實作下皆成立，且 sequence 前綴保證跨序列無
檔名碰撞，最穩健。

純檔案操作，不需重跑模型推論。

Usage
-----
    conda run -n sam_env python scripts/eval/repack_dz_test_submission.py \
        --src  submissions/dz_test_R7_seed42 \
        --out  submissions/dz_test_R7_seed42_submit
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

NUM_CLASSES = 19
DZ_NATIVE_HW = (1080, 1920)
CONF_FULL = 65535  # uint16 飽和值 = 信心 1.0

MODALITIES = ('labelTrainIds', 'confidence', 'labelTrainIds_invalid')


def collect_label_pngs(src: Path) -> list[Path]:
    """蒐集源目錄下所有 labelTrainIds 預測 PNG（遞迴）。"""
    pngs = sorted(src.rglob('*.png'))
    if not pngs:
        raise SystemExit(f"❌ 源目錄無任何 PNG：{src}")
    # 扁平打包前先確認 basename 無碰撞（跨序列）
    names = [p.name for p in pngs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"❌ 偵測到重複 basename，扁平佈局會衝突：{sorted(dup)[:5]}")
    return pngs


def build_submission(src: Path, out: Path) -> int:
    """產生三個子目錄並回傳每目錄的檔案數（應為 151）。"""
    for m in MODALITIES:
        (out / m).mkdir(parents=True, exist_ok=True)

    pngs = collect_label_pngs(src)
    for p in pngs:
        label = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise IOError(f"無法讀取：{p}")
        if label.dtype != np.uint8 or label.shape != DZ_NATIVE_HW:
            raise ValueError(f"非預期格式 dtype={label.dtype} shape={label.shape}：{p}")
        uniq = set(np.unique(label).tolist())
        if not uniq.issubset(set(range(NUM_CLASSES))):
            raise ValueError(f"labelTrainIds 含非法值 {uniq - set(range(NUM_CLASSES))}：{p}")

        name = p.name  # 三個子目錄沿用相同 basename，確保 glob 唯一匹配

        # 1) labelTrainIds：原樣輸出
        _imwrite(out / 'labelTrainIds' / name, label)

        # 2) labelTrainIds_invalid：複製，不標任何 255 → UIoU ≡ IoU
        _imwrite(out / 'labelTrainIds_invalid' / name, label)

        # 3) confidence：常數 65535 的 uint16 圖，尺寸對齊
        conf = np.full(label.shape, CONF_FULL, dtype=np.uint16)
        _imwrite(out / 'confidence' / name, conf)

    return len(pngs)


def _imwrite(path: Path, arr: np.ndarray) -> None:
    if not cv2.imwrite(str(path), arr):
        raise IOError(f"cv2.imwrite 失敗：{path}")


def verify(out: Path, expected: int) -> None:
    """逐模態 sanity check：數量、dtype、值域。"""
    bad = 0
    for m in MODALITIES:
        files = sorted((out / m).glob('*.png'))
        if len(files) != expected:
            print(f"❌ {m}/ 檔案數 {len(files)} ≠ {expected}")
            bad += 1
            continue
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None or img.shape != DZ_NATIVE_HW:
                print(f"❌ {m}: 讀取或尺寸異常：{f}")
                bad += 1
                break
            if m == 'confidence':
                if img.dtype != np.uint16:
                    print(f"❌ confidence dtype={img.dtype}（須 uint16）：{f}")
                    bad += 1
                    break
            else:
                allowed = set(range(NUM_CLASSES)) | ({255} if m.endswith('invalid') else set())
                if img.dtype != np.uint8 or not set(np.unique(img).tolist()).issubset(allowed):
                    print(f"❌ {m}: dtype/值域異常：{f}")
                    bad += 1
                    break
    if bad == 0:
        print(f"✅ Sanity check 通過：三模態各 {expected} 檔，格式符合 CodaLab 23553 規範")
    else:
        print(f"⚠️  {bad} 項不符規範")


def make_zip(out: Path, zip_path: Path) -> None:
    """打包，arcname 以三個 modality 子目錄為頂層。"""
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for m in MODALITIES:
            for png in sorted((out / m).glob('*.png')):
                zf.write(png, arcname=str(Path(m) / png.name))
    print(f"📦 已打包：{zip_path}（{zip_path.stat().st_size / 1e6:.1f} MB）")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--src', type=str,
                   default=str(repo / 'submissions' / 'dz_test_R7_seed42'),
                   help='既有單模態 labelTrainIds 預測根目錄')
    p.add_argument('--out', type=str,
                   default=str(repo / 'submissions' / 'dz_test_R7_seed42_submit'),
                   help='三模態提交輸出根目錄（會額外產生同名 .zip）')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    print(f"Source : {src}")
    print(f"Output : {out}")
    n = build_submission(src, out)
    print(f"✍️  每模態寫出 {n} 檔")
    verify(out, expected=n)
    zip_path = out.with_suffix('.zip')
    make_zip(out, zip_path)


if __name__ == '__main__':
    main()
