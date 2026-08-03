"""PairSAM 在 MUSES 上的推論：val 本地 mIoU + test 提交打包。

固定使用 FULL_seed42 權重（使用者定案）。兩種子任務:

* ``--split val``  : 對 250 張 val（有 GT）推論，於原生解析度 1080×1920 計算
                     標準 mIoU（整體 + 分天氣 + 分日夜）。不產 zip。
* ``--split test`` : 對 750 張 test 推論，輸出 trainIds PNG 並打包成
                     MUSES evaluation server 合規 zip（頂層 ``labelTrainIds/``，
                     檔名 ``REC####_frame_######.png``，尺寸 1920×1080）。

condition_id 處理（--cond-mode）:

* ``off`` : model.use_cond=False，所有樣本 condition_id 走共享索引 0。
            最乾淨、避開 ACDC↔MUSES 分類法不符（reference 仍照常使用）。
* ``map`` : 保留 condition。night→3；白天 fog→0/rain→1/snow→2；clear→0。
            適用 4 條件（ACDC 訓練）模型的零樣本評估。
* ``csv`` : 沿用 CSV 既有的 condition_id，不做任何重新映射。8 條件模型
            （MUSES weather×time_of_day 全交叉，見 make_muses_csv.py --cond-mode
            map8）必須用此項；映射表只在產生端維護一份，避免分歧。

輸出前向與 ACDC 提交腳本一致（1024 推論 → logits 雙線性上採至原生 → argmax），
確保跨資料集評估協定相同。

用法
----
    # val 本地 mIoU（先確認管線 + 看零樣本分數）
    conda run -n sam_env python scripts/eval/dump_muses_preds.py \
        --split val --cond-mode off \
        --csv ../Datasets/muses_ref_rgb_val.csv

    # test 提交打包
    conda run -n sam_env python scripts/eval/dump_muses_preds.py \
        --split test --cond-mode off \
        --csv ../Datasets/muses_ref_rgb_test.csv --zip
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))  # 讓 _eval_common 可被 import
from _eval_common import (  # noqa: E402
    load_pair_sam_from_ablation, make_batched_input,
)

_SEGANY_ROOT = _THIS.parents[2]
if str(_SEGANY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGANY_ROOT))
from torch.utils.data import DataLoader  # noqa: E402
from utils.pair_dataloader import PairSegmentationDataset  # noqa: E402

NUM_CLASSES = 19
IGNORE_INDEX = 255
MUSES_NATIVE_HW = (1080, 1920)  # MUSES frame camera：1920×1080 → (H, W)

# 固定權重（使用者定案）
DEFAULT_CKPT = str(_SEGANY_ROOT / 'outputs_ablation_m2f' / 'FULL_seed42'
                   / 'weather_sam_best_latest.pth')
DEFAULT_OUT_DIR = _SEGANY_ROOT / 'submissions' / 'muses_test'

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]

# ── condition_id 映射（--cond-mode map 用）─────────────────────────
_WEATHER_TO_ID = {'fog': 0, 'rain': 1, 'snow': 2, 'clear': 0}


def resolve_condition_id(weather: str, time_of_day: str) -> int:
    """map 模式：night 一律 3；白天依天氣，clear 落 0（中性回退）。"""
    if str(time_of_day).lower() == 'night':
        return 3
    return _WEATHER_TO_ID.get(str(weather).lower(), 0)


def resolve_condition_csv(src_csv: str, cond_mode: str) -> str:
    """把 condition_id 留白的 CSV 補齊後另存暫存檔，供 dataloader 使用。

    dataloader 對 condition_id 有嚴格檢查（必須 ∈ {0,1,2,3}），故先在此回填:
      * off : 全部填 0（模型端另以 use_cond=False 忽略）
      * map : 依 weather/time_of_day 映射
    回傳暫存 CSV 路徑。原始 CSV（condition_id 留白）保持不動。
    """
    df = pd.read_csv(src_csv)
    if cond_mode == 'csv':
        # CSV 自帶 condition_id（例：make_muses_csv.py --cond-mode map8 產出的 8 類），
        # 直接沿用，避免此處再維護一份會與產生端分歧的映射表。
        if 'condition_id' not in df.columns or df['condition_id'].isna().any():
            raise ValueError(
                f'--cond-mode csv 需要 CSV 的 condition_id 欄位完整填值：{src_csv}')
        return src_csv
    if cond_mode == 'off':
        df['condition_id'] = 0
    elif cond_mode == 'map':
        if not {'weather', 'time_of_day'}.issubset(df.columns):
            raise ValueError("--cond-mode map 需要 CSV 具備 weather / time_of_day 欄位")
        df['condition_id'] = [
            resolve_condition_id(w, t) for w, t in zip(df['weather'], df['time_of_day'])
        ]
    else:
        raise ValueError(f"未知 cond_mode: {cond_mode}")

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_muses_resolved.csv', delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def build_loader(csv_path: str, mode: str, num_workers: int,
                 num_conditions: int = 4) -> DataLoader:
    ds = PairSegmentationDataset(
        csv_file=csv_path, image_size=1024, mode=mode, force_raw_images=True,
        num_conditions=num_conditions,
    )
    return DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=PairSegmentationDataset.collate_fn,
    )


@torch.no_grad()
def predict_native(model, batch: dict, device: str,
                   target_hw: tuple[int, int]) -> np.ndarray:
    """前向：1024 推論 → 語義 logits 上採至原生解析度 → argmax（trainIds）。

    與 dump_acdc_test_preds.predict_native 的 m2f 路徑一致:m2f 語義輸出即
    low_res_logits(19 通道),use_lrh=False 不經 context_fusion_head。
    """
    batched_input = make_batched_input(batch, device)
    outputs = model(batched_input)

    if getattr(model, 'decoder_arch', None) == 'm2f':
        sem = outputs[0]['low_res_logits']            # (1, 19, 256, 256)
        sem_hr = F.interpolate(sem, size=target_hw, mode='bilinear',
                               align_corners=False)
        return sem_hr.argmax(dim=1).squeeze(0).cpu().numpy()

    # 非 m2f 後備路徑（本 checkpoint 為 m2f，理論上不會走到）
    low_res = outputs[0]['low_res_logits'].squeeze(0)
    class_ids = outputs[0]['class_ids']
    full = torch.full((1, NUM_CLASSES, 256, 256), -10.0,
                      device=device, dtype=low_res.dtype)
    for k, c in enumerate(class_ids):
        full[0, c] = low_res[k]
    fused = model.context_fusion_head(full)
    fused_hr = F.interpolate(fused, size=target_hw, mode='bilinear',
                             align_corners=False)
    return fused_hr.argmax(dim=1).squeeze(0).cpu().numpy()


# ── mIoU 累積（標準 Jaccard，忽略 255）───────────────────────────
def update_confmat(cm: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    """把單張 (gt, pred) 累加到 19×19 混淆矩陣（忽略 GT==255）。"""
    valid = (gt != IGNORE_INDEX) & (gt < NUM_CLASSES)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    p = np.clip(p, 0, NUM_CLASSES - 1)
    idx = NUM_CLASSES * g + p
    cm += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(
        NUM_CLASSES, NUM_CLASSES)


def miou_from_confmat(cm: np.ndarray) -> tuple[float, np.ndarray]:
    """由混淆矩陣算 per-class IoU 與 mIoU（僅對 union>0 的類別平均）。"""
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    nz = union > 0
    iou[nz] = inter[nz] / union[nz]
    miou = float(np.nanmean(iou)) if nz.any() else 0.0
    return miou, iou


def run_val(model, loader, df, device: str, max_samples: int | None = None) -> None:
    """val：原生解析度計算整體 + 分天氣 + 分日夜 mIoU。"""
    cm_all = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_weather: dict[str, np.ndarray] = {}
    cm_tod: dict[str, np.ndarray] = {}

    total = min(max_samples, len(df)) if max_samples is not None else len(df)
    n_done = 0
    for idx, batch in enumerate(tqdm(loader, desc='MUSES val', total=total)):
        if max_samples is not None and idx >= max_samples:
            break
        n_done += 1
        row = df.iloc[idx]
        gt = cv2.imread(str(row['gt_path']), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise IOError(f"讀不到 GT：{row['gt_path']}")
        H, W = gt.shape[:2]
        pred = predict_native(model, batch, device, target_hw=(H, W))

        update_confmat(cm_all, gt, pred)
        w = str(row['weather']); t = str(row['time_of_day'])
        cm_weather.setdefault(w, np.zeros_like(cm_all))
        cm_tod.setdefault(t, np.zeros_like(cm_all))
        update_confmat(cm_weather[w], gt, pred)
        update_confmat(cm_tod[t], gt, pred)

    miou, iou = miou_from_confmat(cm_all)
    print('\n' + '=' * 60)
    print(f'MUSES val 整體 mIoU: {miou * 100:.2f}  (n={n_done})')
    print('-' * 60)
    print('分天氣:')
    for w in sorted(cm_weather):
        m, _ = miou_from_confmat(cm_weather[w])
        print(f'  {w:6s}: {m * 100:.2f}')
    print('分日夜:')
    for t in sorted(cm_tod):
        m, _ = miou_from_confmat(cm_tod[t])
        print(f'  {t:6s}: {m * 100:.2f}')
    print('-' * 60)
    print('Per-class IoU:')
    for c in range(NUM_CLASSES):
        v = iou[c]
        print(f'  {CITYSCAPES_CLASSES[c]:14s}: '
              f'{"n/a" if np.isnan(v) else f"{v * 100:.2f}"}')
    print('=' * 60)


def submission_name(image_path: str) -> str:
    """由 frame_camera 檔名切出提交檔名：REC####_frame_######.png。"""
    stem = Path(image_path).name.replace('_frame_camera.png', '')
    return f'{stem}.png'


def run_test(model, loader, df, device: str, out_root: Path,
             do_zip: bool, max_samples: int | None) -> None:
    """test：輸出 trainIds PNG（原生 1080×1920）並可打包 zip。"""
    out_root.mkdir(parents=True, exist_ok=True)
    # 清空舊 PNG：zip 會遞迴打包 out_root 內所有 *.png，殘留舊檔會破壞
    # 「恰 750 檔」的提交規範，故每次執行前先清乾淨。
    stale = list(out_root.glob('*.png'))
    if stale:
        print(f'🧹 清除 {len(stale)} 個既有 PNG（確保提交檔數正確）')
        for p in stale:
            p.unlink()
    written: list[Path] = []
    for idx, batch in enumerate(tqdm(loader, desc='MUSES test', total=len(df))):
        if max_samples is not None and idx >= max_samples:
            break
        pred = predict_native(model, batch, device, target_hw=MUSES_NATIVE_HW)
        pred_u8 = pred.astype(np.uint8)

        out_path = out_root / submission_name(str(df.iloc[idx]['image_path']))
        if not cv2.imwrite(str(out_path), pred_u8):
            raise IOError(f'cv2.imwrite 失敗：{out_path}')
        written.append(out_path)

    verify_outputs(written, expected=(max_samples or len(df)))
    if do_zip:
        make_zip(out_root, out_root.with_suffix('.zip'))


def verify_outputs(written: list[Path], expected: int) -> None:
    """數量 / shape / dtype / 值域 sanity check。"""
    if len(written) != expected:
        print(f'⚠️  輸出 {len(written)} 張 ≠ 預期 {expected}')
    bad = 0
    allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    for p in written:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape != MUSES_NATIVE_HW or img.dtype != np.uint8:
            print(f'❌ 尺寸/dtype 異常：{p} '
                  f'({None if img is None else (img.shape, img.dtype)})')
            bad += 1
            continue
        uniq = set(np.unique(img).tolist())
        if not uniq.issubset(allowed):
            print(f'❌ 含非法 trainId {uniq - allowed}：{p}')
            bad += 1
    print(f'{"✅" if bad == 0 else "⚠️"}  Sanity check：{len(written) - bad}/{len(written)} 合規')


def make_zip(out_root: Path, zip_path: Path) -> None:
    """打包為 MUSES 合規 zip：頂層 labelTrainIds/，檔案扁平置於其下。"""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for png in sorted(out_root.glob('*.png')):
            zf.write(png, arcname=str(Path('labelTrainIds') / png.name))
            n += 1
    size_mb = zip_path.stat().st_size / 1e6
    flag = '⚠️ 超過 200MB!' if size_mb > 200 else ''
    print(f'📦 已打包：{zip_path}（{size_mb:.1f} MB, {n} files）{flag}')


def parse_args() -> argparse.Namespace:
    repo_root = _SEGANY_ROOT.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--split', required=True, choices=['val', 'test'])
    p.add_argument('--cond-mode', required=True, choices=['off', 'map', 'csv'],
                   help='condition_id 處理：off=關閉(use_cond=False)；'
                        'map=依天氣/日夜映射為 4 類；'
                        'csv=沿用 CSV 既有的 condition_id（8 類 map8 模型用此項）')
    p.add_argument('--csv', type=str, default=None,
                   help='預設依 --split 取 Datasets/muses_ref_rgb_{split}.csv')
    p.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    p.add_argument('--out', type=str, default=str(DEFAULT_OUT_DIR),
                   help='test 提交輸出根目錄')
    p.add_argument('--zip', action='store_true', help='test 完成後打包 zip')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=None, help='只跑前 N 張(debug)')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.csv is None:
        args.csv = str(repo_root / 'Datasets' / f'muses_ref_rgb_{args.split}.csv')
    return args


def main() -> None:
    args = parse_args()
    print(f'Checkpoint : {args.ckpt}')
    print(f'CSV        : {args.csv}')
    print(f'Split      : {args.split} | cond-mode: {args.cond_mode} | device: {args.device}')

    model, _cfg = load_pair_sam_from_ablation(args.ckpt, device=args.device)
    if args.cond_mode == 'off':
        model.use_cond = False   # 忽略 condition_id（走共享索引 0），reference 仍使用
        print('  → use_cond=False（condition 關閉）')

    resolved_csv = resolve_condition_csv(args.csv, args.cond_mode)
    # dataloader 的 condition_id 上界必須與模型 embedding 列數一致，否則 8 類 CSV
    # 會在 4 條件檢查下被擋（或反之，越界索引直接查表出錯）。
    loader = build_loader(resolved_csv, mode=args.split, num_workers=args.num_workers,
                          num_conditions=getattr(model, 'num_conditions', 4))
    df = loader.dataset.data.reset_index(drop=True)

    if args.split == 'val':
        run_val(model, loader, df, args.device, max_samples=args.max_samples)
    else:
        run_test(model, loader, df, args.device,
                 out_root=Path(args.out), do_zip=args.zip,
                 max_samples=args.max_samples)


if __name__ == '__main__':
    main()
