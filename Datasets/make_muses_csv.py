"""從 MUSES meta.json 產生推論用 / 訓練用 CSV。

輸出 CSV 欄位（相容 PairSegmentationDataset，並多帶 weather/time_of_day 供
condition_id 映射與分層分析）:

    image_path, ref_image_path, gt_path, condition, condition_id,
    weather, time_of_day, invalid_mask

設計要點
--------
1. **權威來源是 meta.json**：它逐張列出 split / weather / time_of_day 以及
   ``reference_frame_available``，避免用檔案系統掃描猜測。
2. **clear-day 自身當參考**（使用者定案 B）：meta 中 ``reference_frame_available == 'No'``
   者僅 clear-day，這類樣本 ``ref_image_path`` 直接指向自身 frame_camera 影像
   （語意正確：晴天日間本就是乾淨參考，UAWarpC 對自身近似 identity 對齊）。
   ``--adverse-only`` 會整批排除這些樣本，對齊 ACDC adverse CSV 的組成。
3. **condition_id**：``--cond-mode blank``（預設，向後相容）留白；``map`` 依
   ACDC 慣例回填 fog=0 / rain=1 / snow=2 / night=3（夜間一律 3，覆蓋天氣）；
   ``map8`` 展開為 weather×time_of_day 全交叉 8 類，前 4 格語意與 ACDC 相同，
   使 ACDC checkpoint 的 condition embedding 可原位沿用。
   dataloader 對此欄為 strict 檢查（須落在 ``[0, num_conditions)``）。
4. **gt 僅 train/val 有**：gt_semantic_trainval 不含 test；test 列 ``gt_path`` 留空。
   語意 GT 取 ``_gt_labelTrainIds.png``（trainIds 0..18，255=ignore），與 dataloader
   直接以 trainIds 讀取的約定一致。
5. **invalid_mask 留空**：MUSES 無此標註（dataloader 會回退為全 False）。

用法
----
    # 推論用（既有行為，condition_id 留白）
    python Datasets/make_muses_csv.py --splits val test

    # 訓練用（ACDC 對齊：僅惡劣條件 + 4 類 condition_id）
    python Datasets/make_muses_csv.py \
        --splits train val --adverse-only --cond-mode map \
        --out-prefix muses_adverse_ref_rgb --verify-paths

    # 訓練用（8 類 weather×time_of_day 全交叉，含 clear-day）
    python Datasets/make_muses_csv.py \
        --splits train val --cond-mode map8 \
        --out-prefix muses_cond8_ref_rgb --verify-paths
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


# ── condition_id 映射 ──────────────────────────────────────────────
# map（4 類，ACDC 相容）：fog=0, rain=1, snow=2, night=3。
# MUSES 是 weather × time_of_day 的二維交叉，ACDC 則是單一條件標籤；4 類方案下
# 夜間樣本一律歸 3（覆蓋天氣），白天樣本依天氣歸類。
_WEATHER_TO_ID = {'fog': 0, 'rain': 1, 'snow': 2}

# map8（8 類，weather × time_of_day 全交叉）：
# 前 4 格的語意與 ACDC 完全相同，使 ACDC checkpoint 的 condition embedding 可原位
# 沿用（見 build_pair_sam._expand_condition_embedding）；後 4 格為 MUSES 新增組合。
_COND8 = {
    ('fog',   'day'):   ('fog',         0),
    ('rain',  'day'):   ('rain',        1),
    ('snow',  'day'):   ('snow',        2),
    ('clear', 'night'): ('clear_night', 3),   # ≡ ACDC 的 night（晴天夜間）
    ('fog',   'night'): ('fog_night',   4),
    ('rain',  'night'): ('rain_night',  5),
    ('snow',  'night'): ('snow_night',  6),
    ('clear', 'day'):   ('clear_day',   7),
}


def resolve_condition(weather: str, tod: str) -> tuple[str, int]:
    """4 類映射，與 dump_muses_preds.resolve_condition_id 同義。"""
    if str(tod).lower() == 'night':
        return 'night', 3
    w = str(weather).lower()
    if w not in _WEATHER_TO_ID:
        # clear-day：ACDC 無對應條件，落中性 0（僅在未加 --adverse-only 時出現）
        return w, 0
    return w, _WEATHER_TO_ID[w]


def resolve_condition8(weather: str, tod: str) -> tuple[str, int]:
    """8 類映射：weather × time_of_day 全交叉，前 4 格保持 ACDC 語意。"""
    key = (str(weather).lower(), str(tod).lower())
    if key not in _COND8:
        raise ValueError(f'未預期的 weather/time_of_day 組合：{key}')
    return _COND8[key]


def build_rows(meta: dict, muses_root: Path, split: str,
               adverse_only: bool = False, cond_mode: str = 'blank') -> list[dict]:
    """為指定 split 產生所有樣本列（依 image id 排序，穩定可重現）。"""
    rows: list[dict] = []
    for img_id in sorted(meta.keys()):
        rec = meta[img_id]
        if rec.get('split') != split:
            continue

        frame_rel = rec['path_to_frame_camera']            # frame_camera/{split}/{w}/{i}/...
        weather = rec['weather']                            # clear / fog / rain / snow
        tod = rec['time_of_day']                            # day / night

        has_ref = rec.get('reference_frame_available') == 'Yes'

        # --adverse-only：排除 clear-day（無官方參考、非惡劣條件），對齊 ACDC 組成
        if adverse_only and not has_ref:
            continue

        image_path = str(muses_root / frame_rel)

        # 參考影像：有則用官方參考，clear-day 無參考則自身當參考
        if has_ref:
            ref_path = str(muses_root / rec['path_to_reference_frame'])
        else:
            ref_path = image_path                           # 自身當參考（定案 B）

        # GT：僅 train/val 有；test 留空
        gt_path = ''
        if split in ('train', 'val'):
            gt_path = str(muses_root / frame_rel_to_gt_rel(frame_rel))

        # condition / condition_id：blank 保留舊行為；map=4 類、map8=8 類全交叉
        if cond_mode == 'map':
            cond_label, cond_id = resolve_condition(weather, tod)
        elif cond_mode == 'map8':
            cond_label, cond_id = resolve_condition8(weather, tod)
        else:
            cond_label, cond_id = weather, ''   # per-weather mIoU 標籤，id 留白

        rows.append({
            'image_path': image_path,
            'ref_image_path': ref_path,
            'gt_path': gt_path,
            'condition': cond_label,
            'condition_id': cond_id,
            'weather': weather,            # 原始天氣（分層分析用，不受 cond_mode 影響）
            'time_of_day': tod,            # 原始時段（分層分析用）
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
    p.add_argument('--adverse-only', action='store_true',
                   help='排除 clear-day（無官方參考影像），使組成對齊 ACDC adverse CSV')
    p.add_argument('--cond-mode', choices=['blank', 'map', 'map8'], default='blank',
                   help='condition_id：blank=留白（推論用，預設）；'
                        'map=4 類 fog0/rain1/snow2/night3（ACDC 相容）；'
                        'map8=8 類 weather×time_of_day 全交叉（訓練用，需 '
                        '--num_conditions 8）')
    p.add_argument('--out-prefix', type=str, default='muses_ref_rgb',
                   help='輸出檔名前綴，最終為 {prefix}_{split}.csv')
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
        rows = build_rows(meta, muses_root, split,
                          adverse_only=args.adverse_only, cond_mode=args.cond_mode)
        out_path = Path(args.out_dir) / f'{args.out_prefix}_{split}.csv'
        write_csv(rows, out_path)

        # 統計摘要
        n_selfref = sum(1 for r in rows if r['ref_image_path'] == r['image_path'])
        n_gt = sum(1 for r in rows if r['gt_path'])
        print(f'[{split}] {len(rows)} 列 → {out_path}')
        print(f'        自身當參考(clear-day): {n_selfref}；有 GT: {n_gt}')

        # condition 分布（訓練前確認類別平衡）
        dist: dict[tuple, int] = {}
        for r in rows:
            key = (r['condition'], r['condition_id'], r['weather'], r['time_of_day'])
            dist[key] = dist.get(key, 0) + 1
        for (cond, cid, w, t), n in sorted(dist.items()):
            print(f'        condition={cond:<6} id={str(cid):<2} '
                  f'({w}/{t}): {n}')

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
