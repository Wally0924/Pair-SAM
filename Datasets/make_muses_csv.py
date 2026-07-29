"""從 MUSES meta.json 產生 val / test 的推論用 CSV。

輸出 CSV 欄位（相容 PairSegmentationDataset，並多帶 weather/time_of_day 供日後
決定 condition_id 映射時一鍵回填）:

    image_path, ref_image_path, gt_path, condition, condition_id,
    weather, time_of_day, invalid_mask

設計要點
--------
1. **權威來源是 meta.json**：它逐張列出 split / weather / time_of_day 以及
   ``reference_frame_available``，避免用檔案系統掃描猜測。
2. **clear-day 自身當參考**（使用者定案 B）：meta 中 ``reference_frame_available == 'No'``
   者僅 clear-day，這類樣本 ``ref_image_path`` 直接指向自身 frame_camera 影像
   （語意正確：晴天日間本就是乾淨參考，UAWarpC 對自身近似 identity 對齊）。
3. **condition_id 先留空**（使用者定案 A，延後決定）：CSV 該欄留白，待映射策略
   確定後由 dump_muses_preds.py 依 --cond-mode 於執行時回填，或另跑回填腳本。
4. **gt 僅 train/val 有**：gt_semantic_trainval 不含 test；test 列 ``gt_path`` 留空。
   語意 GT 取 ``_gt_labelTrainIds.png``（trainIds 0..18，255=ignore），與 dataloader
   直接以 trainIds 讀取的約定一致。
5. **invalid_mask 留空**：MUSES 無此標註。

用法
----
    python Datasets/make_muses_csv.py \
        --muses-root /home/rvl1421/Datasets/MUSES \
        --out-dir    Datasets \
        --splits val test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def frame_rel_to_gt_rel(frame_rel: str) -> str:
    """由 frame_camera 相對路徑推導 gt_semantic(labelTrainIds) 相對路徑。

    frame_camera/val/fog/day/REC0475_frame_000247_frame_camera.png
      -> gt_semantic/val/fog/day/REC0475_frame_000247_gt_labelTrainIds.png
    """
    gt_rel = frame_rel.replace('frame_camera/', 'gt_semantic/', 1)
    gt_rel = gt_rel.replace('_frame_camera.png', '_gt_labelTrainIds.png')
    return gt_rel


def build_rows(meta: dict, muses_root: Path, split: str) -> list[dict]:
    """為指定 split 產生所有樣本列（依 image id 排序，穩定可重現）。"""
    rows: list[dict] = []
    for img_id in sorted(meta.keys()):
        rec = meta[img_id]
        if rec.get('split') != split:
            continue

        frame_rel = rec['path_to_frame_camera']            # frame_camera/{split}/{w}/{i}/...
        weather = rec['weather']                            # clear / fog / rain / snow
        tod = rec['time_of_day']                            # day / night

        image_path = str(muses_root / frame_rel)

        # 參考影像：有則用官方參考，clear-day 無參考則自身當參考
        if rec.get('reference_frame_available') == 'Yes':
            ref_path = str(muses_root / rec['path_to_reference_frame'])
        else:
            ref_path = image_path                           # 自身當參考（定案 B）

        # GT：僅 train/val 有；test 留空
        gt_path = ''
        if split in ('train', 'val'):
            gt_path = str(muses_root / frame_rel_to_gt_rel(frame_rel))

        rows.append({
            'image_path': image_path,
            'ref_image_path': ref_path,
            'gt_path': gt_path,
            'condition': weather,          # per-weather mIoU 標籤
            'condition_id': '',            # 先留空（定案 A，延後決定映射）
            'weather': weather,            # 供日後回填 condition_id
            'time_of_day': tod,            # 供日後回填 condition_id
            'invalid_mask': '',            # MUSES 無此標註
        })
    return rows


FIELDNAMES = [
    'image_path', 'ref_image_path', 'gt_path', 'condition', 'condition_id',
    'weather', 'time_of_day', 'invalid_mask',
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--muses-root', type=str, default='/home/rvl1421/Datasets/MUSES',
                   help='MUSES 解壓後根目錄（含 meta.json 與 frame_camera/ 等）')
    p.add_argument('--out-dir', type=str,
                   default=str(Path(__file__).resolve().parent),
                   help='CSV 輸出目錄（預設為本 Datasets/ 目錄）')
    p.add_argument('--splits', nargs='+', default=['val', 'test'],
                   choices=['train', 'val', 'test'])
    p.add_argument('--verify-paths', action='store_true',
                   help='逐列檢查 image/ref/gt 檔案是否存在（解壓完成後建議開啟）')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    muses_root = Path(args.muses_root)
    meta_path = muses_root / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'找不到 meta.json：{meta_path}（是否尚未解壓？）')

    with open(meta_path) as f:
        meta = json.load(f)
    print(f'meta.json 載入：{len(meta)} 張影像')

    for split in args.splits:
        rows = build_rows(meta, muses_root, split)
        out_path = Path(args.out_dir) / f'muses_ref_rgb_{split}.csv'
        write_csv(rows, out_path)

        # 統計摘要
        n_selfref = sum(1 for r in rows if r['ref_image_path'] == r['image_path'])
        n_gt = sum(1 for r in rows if r['gt_path'])
        print(f'[{split}] {len(rows)} 列 → {out_path}')
        print(f'        自身當參考(clear-day): {n_selfref}；有 GT: {n_gt}')

        if args.verify_paths:
            missing = 0
            for r in rows:
                for col in ('image_path', 'ref_image_path'):
                    if not Path(r[col]).exists():
                        missing += 1
                        if missing <= 5:
                            print(f'        ❌ 缺檔 {col}: {r[col]}')
                if r['gt_path'] and not Path(r['gt_path']).exists():
                    missing += 1
                    if missing <= 5:
                        print(f'        ❌ 缺檔 gt_path: {r["gt_path"]}')
            print(f'        路徑檢查：{"✅ 全部存在" if missing == 0 else f"⚠️ {missing} 個缺檔"}')


if __name__ == '__main__':
    main()
