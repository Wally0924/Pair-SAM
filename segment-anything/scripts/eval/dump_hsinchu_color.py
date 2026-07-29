"""PairSAM 對 hsinchu_frames 純推論：輸出 Cityscapes 19 類彩色分割圖。

不計算任何指標(該資料集無 GT),只把每張 frame 的語義預測上色後存檔。

固定使用 FULL_seed42 權重(與 MUSES 零樣本協定同一顆)。推論協定(使用者定案):

* **自身當參考**:hsinchu_frames 無對應晴天參考影像,故 ref_image = 輸入自身
  (與 MUSES clear-day 自參考同理,CrossViewAlignment 對自身近似 identity)。
* **cond 關閉**:model.use_cond=False,所有樣本走共享索引 0,避開跨資料集
  天氣分類法不符;reference 仍照常使用。

前向與 MUSES/ACDC 提交腳本一致:1024 推論 → 語義 logits 雙線性上採至目標解析度
→ argmax(trainIds) → colorize_19class 上色。

輸出:
* 目標解析度 1080×1920(使用者定案;原生為 2160×3840,等比 2×)。
* 檔名 ``{stem}_color.png``(對齊 Cityscapes/ACDC 的 color image 命名)。
* PNG 以 BGR 寫出(cv2 慣例),色盤為 Cityscapes 19 類標準色。

用法
----
    conda run -n sam_env python scripts/eval/dump_hsinchu_color.py \
        --frames-dir /home/rvl1421/Datasets/hsinchu_frames \
        --out outputs_hsinchu_color
"""
from __future__ import annotations

import argparse
import sys
import tempfile
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
    colorize_19class, load_pair_sam_from_ablation, make_batched_input,
)

_SEGANY_ROOT = _THIS.parents[2]
if str(_SEGANY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGANY_ROOT))
from torch.utils.data import DataLoader  # noqa: E402
from utils.pair_dataloader import PairSegmentationDataset  # noqa: E402

# 固定權重(使用者定案):FULL_seed42
DEFAULT_CKPT = str(_SEGANY_ROOT / 'outputs_ablation_m2f' / 'FULL_seed42'
                   / 'weather_sam_best_latest.pth')
DEFAULT_FRAMES_DIR = '/home/rvl1421/Datasets/hsinchu_frames'
DEFAULT_OUT_DIR = str(_SEGANY_ROOT / 'outputs_hsinchu_color')
# 預設 None → 逐幀輸出各自「原生解析度」(邊界配準最精細);
# 可用 --height/--width 覆寫成固定尺寸。

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def build_frames_csv(frames_dir: Path,
                     start: int | None = None, end: int | None = None) -> str:
    """掃描資料夾內所有影像,產生 PairSegmentationDataset 相容的暫存 CSV。

    * image_path / ref_image_path 皆指向 frame 自身(自參考)。
    * condition_id 一律 0(cond 關閉,dataloader 仍要求 ∈ {0,1,2,3})。
    * gt_path / invalid_mask 留空(無 GT、無盲區標註)。
    * start/end:依「排序後 0-based 索引」切片(兩端皆含),對應 GPS 路段幀範圍。
    回傳暫存 CSV 路徑(依檔名排序,穩定可重現)。
    """
    frames = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    )
    if not frames:
        raise FileNotFoundError(f'資料夾內找不到影像:{frames_dir}')
    if start is not None or end is not None:
        s = 0 if start is None else start
        e = len(frames) - 1 if end is None else end   # end 含
        frames = frames[s:e + 1]
        if not frames:
            raise ValueError(f'幀範圍 [{s}, {e}] 切出 0 張,請檢查 --start/--end')

    df = pd.DataFrame({
        'image_path': [str(p) for p in frames],
        'ref_image_path': [str(p) for p in frames],  # 自身當參考
        'gt_path': '',
        'condition_id': 0,                            # cond 關閉,填中性 0
        'invalid_mask': '',
    })
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_hsinchu.csv', delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def build_loader(csv_path: str, num_workers: int) -> DataLoader:
    ds = PairSegmentationDataset(
        csv_file=csv_path, image_size=1024, mode='test', force_raw_images=True,
    )
    return DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=PairSegmentationDataset.collate_fn,
    )


@torch.no_grad()
def predict_native(model, batch: dict, device: str,
                   target_hw: tuple[int, int]) -> np.ndarray:
    """前向:1024 推論 → 語義 logits 上採至 target_hw → argmax(trainIds)。

    與 dump_muses_preds.predict_native 的 m2f 路徑一致:m2f 語義輸出即
    low_res_logits(19 通道),use_lrh=False 不經 context_fusion_head。
    """
    batched_input = make_batched_input(batch, device)
    outputs = model(batched_input)
    sem = outputs[0]['low_res_logits']                # (1, 19, 256, 256)
    sem_hr = F.interpolate(sem, size=target_hw, mode='bilinear',
                           align_corners=False)
    return sem_hr.argmax(dim=1).squeeze(0).cpu().numpy()


def run(model, loader, df, device: str, out_root: Path,
        target_hw: tuple[int, int] | None, max_samples: int | None = None) -> None:
    """逐張推論 → 上色 → 存 {stem}_color.png。

    target_hw=None 時,每張以其原生尺寸(batch['original_size'])為上採目標。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    total = min(max_samples, len(df)) if max_samples is not None else len(df)
    n_done = 0
    for idx, batch in enumerate(tqdm(loader, desc='hsinchu color', total=total)):
        if max_samples is not None and idx >= max_samples:
            break
        if target_hw is None:
            H, W = batch['original_size'][0]          # 該幀原生 (H, W)
            tgt = (int(H), int(W))
        else:
            tgt = target_hw
        pred = predict_native(model, batch, device, target_hw=tgt)  # trainIds
        color_rgb = colorize_19class(pred)                                # (H, W, 3) RGB
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)            # cv2 以 BGR 寫檔

        stem = Path(str(df.iloc[idx]['image_path'])).stem
        out_path = out_root / f'{stem}_color.png'
        if not cv2.imwrite(str(out_path), color_bgr):
            raise IOError(f'cv2.imwrite 失敗:{out_path}')
        n_done += 1
    print(f'\n✅ 完成:{n_done} 張彩色分割圖 → {out_root}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--frames-dir', type=str, default=DEFAULT_FRAMES_DIR)
    p.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    p.add_argument('--out', type=str, default=DEFAULT_OUT_DIR)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=None, help='只跑前 N 張(debug)')
    p.add_argument('--start', type=int, default=None,
                   help='起始幀索引(排序後 0-based,含);對應 GPS 路段')
    p.add_argument('--end', type=int, default=None,
                   help='結束幀索引(排序後 0-based,含)')
    p.add_argument('--height', type=int, default=None,
                   help='輸出高;省略則用各幀原生高')
    p.add_argument('--width', type=int, default=None,
                   help='輸出寬;省略則用各幀原生寬')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_hw = (args.height, args.width) if (
        args.height is not None and args.width is not None) else None
    print(f'Checkpoint : {args.ckpt}')
    print(f'Frames dir : {args.frames_dir}')
    print(f'Out dir    : {args.out} | target HW: '
          f'{target_hw or "native(逐幀原生)"} | device: {args.device}')

    csv_path = build_frames_csv(Path(args.frames_dir), start=args.start, end=args.end)
    model, _cfg = load_pair_sam_from_ablation(args.ckpt, device=args.device)
    model.use_cond = False   # cond 關閉(走共享索引 0),reference 仍使用
    print('  → use_cond=False(condition 關閉)、ref=自身')

    loader = build_loader(csv_path, num_workers=args.num_workers)
    df = loader.dataset.data.reset_index(drop=True)
    run(model, loader, df, args.device, Path(args.out), target_hw,
        max_samples=args.max_samples)


if __name__ == '__main__':
    main()
