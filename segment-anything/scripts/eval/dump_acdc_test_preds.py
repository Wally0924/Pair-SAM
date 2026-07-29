"""Dump PairSAM v15 predictions for ACDC test-set submission.

產生符合 ACDC evaluation server 規範的提交檔案：
  * trainIds 0..18（不預測像素填 255），uint8 PNG
  * 原生解析度 1080x1920（不縮放）
  * 檔名：input ``*_rgb_anon.png`` → ``*_gt_labelTrainIds.png``
  * 目錄結構鏡像 GT：``<condition>/test/<sequence>/<file>``
  * 任務：standard semantic segmentation（不是 uncertainty-aware track）

本腳本 forward 邏輯：1024 推論 → fused logits →
雙線性上採至原解析度 → argmax。

Usage
-----
    conda run -n sam_env python scripts/eval/dump_acdc_test_preds.py \
        --csv ../Datasets/acdc_adverse_ref_rgb_test.csv \
        --ckpt outputs_pair_sam_mask2former_testv15/weather_sam_best_latest.pth \
        --out  submissions/acdc_test_v15 \
        --zip
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))  # 讓 _eval_common 可被 import
from _eval_common import (  # noqa: E402
    load_pair_sam_from_ablation, make_batched_input, colorize_19class,
    CITYSCAPES_CLASSES, CITYSCAPES_PALETTE,
)

# segment-anything 根目錄（給 utils import 用，與 _eval_common 同樣的處理）
_SEGANY_ROOT = _THIS.parents[2]
if str(_SEGANY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGANY_ROOT))
from torch.utils.data import DataLoader  # noqa: E402
from utils.pair_dataloader import PairSegmentationDataset  # noqa: E402

NUM_CLASSES = 19
IGNORE_INDEX = 255
ACDC_NATIVE_HW = (1080, 1920)  # ACDC test images 全部為此尺寸

# 固定輸出位置：與 cwd 無關，避免相對路徑被嵌套
DEFAULT_OUT_DIR = _SEGANY_ROOT / 'submissions' / 'acdc_test_v15'
DEFAULT_VIZ_DIR = _SEGANY_ROOT / 'submissions' / 'acdc_test_v15_viz'

# matplotlib 延遲 import：viz 才需要，避免無 viz 時也付出 import 成本
_plt = None
_mpatches = None
_gridspec = None


def _ensure_matplotlib():
    """延遲載入 matplotlib，且固定使用 Agg backend 以支援無顯示環境。"""
    global _plt, _mpatches, _gridspec
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        _plt, _mpatches, _gridspec = plt, mpatches, gridspec


def save_triptych(adverse_rgb: np.ndarray, clear_rgb: np.ndarray | None,
                  pred: np.ndarray, condition: str, sample_id: str,
                  out_path: Path) -> None:
    """三聯圖：Adverse | Clear Reference | Prediction（+ class legend）。

    為 ACDC test 設計（無 GT），即 val 視覺化的 2x2 版本去掉 GT panel。
    """
    _ensure_matplotlib()
    H, W = pred.shape
    # 對齊 adverse/clear 解析度（理論上都是 1080x1920，但保險起見 resize）
    if adverse_rgb.shape[:2] != (H, W):
        adverse_rgb = cv2.resize(adverse_rgb, (W, H), interpolation=cv2.INTER_AREA)
    if clear_rgb is not None and clear_rgb.shape[:2] != (H, W):
        clear_rgb = cv2.resize(clear_rgb, (W, H), interpolation=cv2.INTER_AREA)
    pred_color = colorize_19class(pred)

    fig = _plt.figure(figsize=(24, 10))
    fig.suptitle(f"{sample_id} ({condition})", fontsize=28, fontweight='bold', y=0.98)

    gs = _gridspec.GridSpec(2, 3, height_ratios=[1, 0.18], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(adverse_rgb); ax1.set_title("Input (Adverse)", fontsize=20); ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred_color)
    ax2.set_title("Prediction (PairSAM)", fontsize=20); ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    if clear_rgb is not None:
        ax3.imshow(clear_rgb)
    ax3.set_title("Clear-Weather Reference", fontsize=20); ax3.axis('off')

    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')
    present = sorted(int(v) for v in np.unique(pred) if 0 <= int(v) < NUM_CLASSES)
    patches = [
        _mpatches.Patch(
            color=CITYSCAPES_PALETTE[c] / 255.0,
            label=CITYSCAPES_CLASSES[c],
        )
        for c in present
    ]
    if patches:
        legend = ax_legend.legend(
            handles=patches, loc='center',
            ncol=min(len(patches), 10),
            frameon=False, fontsize=14, title="Classes Present",
        )
        legend.get_title().set_fontsize(16)

    _plt.tight_layout()
    _plt.subplots_adjust(top=0.90)
    _plt.savefig(out_path, dpi=80, bbox_inches='tight')
    _plt.close(fig)


def build_test_loader(csv_path: str, num_workers: int = 4) -> DataLoader:
    """ACDC test loader：mode='test' 會 bypass GT dropna 過濾。"""
    ds = PairSegmentationDataset(
        csv_file=csv_path, image_size=1024, mode='test', force_raw_images=True,
    )
    return DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=PairSegmentationDataset.collate_fn,
    )


@torch.no_grad()
def predict_native(model, batch: dict, device: str, target_hw: tuple[int, int]) -> np.ndarray:
    """前向流程（paper protocol，原生解析度評估）。

    1. forward 在 1024x1024
    2. low_res_logits 攤平回 19 通道（缺席類別填 -10 視同極低）
    3. context_fusion_head → 1024 解析度語義 logits
    4. 雙線性上採至 target_hw（1080x1920）
    5. argmax → trainIds (int64)，呼叫端會 cast 為 uint8
    """
    batched_input = make_batched_input(batch, device)
    outputs = model(batched_input)

    # [M2F] 語義輸出即 low_res_logits(sem_lr,已是 19 通道,class_ids=[0..18])。
    # m2f 的 use_lrh=False,語義推論不經 context_fusion_head(該模組於 e2e/decoder_ft
    # 皆未訓練、為隨機初始化,套用會汙染 logits)。前向對齊 pair_trainer.py val 的
    # m2f mIoU 路徑(postprocess_masks bilinear 上採 → argmax),只是上採到原生解析度。
    if getattr(model, 'decoder_arch', None) == 'm2f':
        sem = outputs[0]['low_res_logits']  # (1, 19, 256, 256)
        sem_hr = F.interpolate(
            sem, size=target_hw, mode='bilinear', align_corners=False,
        )
        return sem_hr.argmax(dim=1).squeeze(0).cpu().numpy()

    low_res = outputs[0]['low_res_logits'].squeeze(0)  # (K, 256, 256)
    class_ids = outputs[0]['class_ids']

    full = torch.full(
        (1, NUM_CLASSES, 256, 256), -10.0,
        device=device, dtype=low_res.dtype,
    )
    for k, c in enumerate(class_ids):
        full[0, c] = low_res[k]

    fused = model.context_fusion_head(full)  # (1, 19, 256, 256)
    fused_hr = F.interpolate(
        fused, size=target_hw, mode='bilinear', align_corners=False,
    )
    return fused_hr.argmax(dim=1).squeeze(0).cpu().numpy()


def submission_relpath(image_path: str) -> Path:
    """從 ACDC input 路徑切出 submission 內的相對路徑。

    範例輸入：
      /home/.../ACDC/rgb_anon/fog/test/GOPR0475/GOPR0475_frame_000247_rgb_anon.png
    輸出：
      fog/test/GOPR0475/GOPR0475_frame_000247_gt_labelTrainIds.png

    以 ``rgb_anon`` 作為錨點切片，避免 ACDC 安裝路徑變動造成相對結構錯亂。
    """
    p = Path(image_path)
    parts = p.parts
    try:
        anchor = parts.index('rgb_anon')
    except ValueError as e:
        raise ValueError(
            f"image_path 不含 'rgb_anon' 錨點：{image_path}。"
            "ACDC 標準佈局需要此資料夾以推導 <condition>/test/<sequence>/...。"
        ) from e
    rel_parts = parts[anchor + 1:]  # ('fog', 'test', 'GOPR0475', '..._rgb_anon.png')
    fname = rel_parts[-1].replace('_rgb_anon.png', '_gt_labelTrainIds.png')
    return Path(*rel_parts[:-1]) / fname


def select_stratified_indices(csv_df, n_per_condition: int) -> set[int]:
    """每個 condition_id (0..3) 取前 N 筆的 row index，回傳合集。

    依 CSV 原始順序選前 N 筆（穩定可重現）；某 condition 不足 N 筆時取全部並警告。
    """
    selected: set[int] = set()
    for cid in (0, 1, 2, 3):
        rows = csv_df.index[csv_df['condition_id'] == cid].tolist()
        if len(rows) < n_per_condition:
            print(f"⚠️  condition_id={cid} 只有 {len(rows)} 筆，少於要求 {n_per_condition}")
        selected.update(rows[:n_per_condition])
    return selected


def dump_predictions(model, loader: DataLoader, csv_df, device: str,
                     out_root: Path, max_samples: int | None = None,
                     viz_root: Path | None = None,
                     viz_indices: set[int] | None = None,
                     viz_only: bool = False) -> list[Path]:
    """主流程：forward → 存 trainId PNG（submission 用）；可選同步存彩色版（debug）。

    viz_root 不為 None 時，會在獨立目錄存 RGB 上色版（套 Cityscapes palette），
    路徑結構與 out_root 一致，但檔名後綴改為 ``_color.png`` 以避免與 submission 混淆。

    Args:
        viz_indices: 若提供，僅這些 CSV row index 會輸出彩色版（其他樣本仍輸出 submission）。
        viz_only: 若為 True，**僅**對 viz_indices 內的樣本跑推論並輸出兩份檔案；
                  其餘樣本完全跳過（不前向、不寫檔），用於快速 debug。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    if viz_root is not None:
        viz_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = len(viz_indices) if viz_only and viz_indices is not None else len(csv_df)
    pbar = tqdm(loader, desc='ACDC test dump', total=total)
    for idx, batch in enumerate(pbar):
        if max_samples is not None and idx >= max_samples:
            break
        if viz_only and viz_indices is not None and idx not in viz_indices:
            continue  # 純 debug 模式：跳過未被抽中的樣本
        pred = predict_native(model, batch, device, target_hw=ACDC_NATIVE_HW)
        # 安全性：trainIds 0..18 已是 argmax 結果範圍，直接 cast；模型理論上不會輸出 255
        pred_u8 = pred.astype(np.uint8)

        rel = submission_relpath(str(csv_df.iloc[idx]['image_path']))
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # PNG 無損；uint8 單通道
        ok = cv2.imwrite(str(out_path), pred_u8)
        if not ok:
            raise IOError(f"cv2.imwrite 失敗：{out_path}")
        written.append(out_path)

        do_viz = viz_root is not None and (viz_indices is None or idx in viz_indices)
        if do_viz:
            # 三聯圖：Adverse | Clear Reference | Prediction（+ class legend）
            # 扁平化路徑：<viz_root>/<condition>/<basename>_triptych.png
            row = csv_df.iloc[idx]
            condition = rel.parts[0]
            basename = rel.stem.replace('_gt_labelTrainIds', '')
            viz_path = viz_root / condition / f"{basename}_triptych.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)

            adverse = cv2.imread(str(row['image_path']), cv2.IMREAD_COLOR)
            adverse = cv2.cvtColor(adverse, cv2.COLOR_BGR2RGB) if adverse is not None else None
            clear = cv2.imread(str(row['ref_image_path']), cv2.IMREAD_COLOR)
            clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB) if clear is not None else None

            save_triptych(
                adverse_rgb=adverse,
                clear_rgb=clear,
                pred=pred_u8,
                condition=str(row.get('condition', condition)),
                sample_id=basename,
                out_path=viz_path,
            )
    return written


def verify_outputs(written: list[Path], expected_count: int | None = None) -> None:
    """快速 sanity check：數量、shape、dtype、值域。對全部 2000 張做檢查。"""
    if expected_count is not None and len(written) != expected_count:
        print(f"⚠️  輸出檔案數 {len(written)} ≠ 預期 {expected_count}")
    bad = 0
    for p in written:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ 無法讀回：{p}")
            bad += 1
            continue
        if img.shape != ACDC_NATIVE_HW:
            print(f"❌ shape={img.shape}（預期 {ACDC_NATIVE_HW}）：{p}")
            bad += 1
            continue
        if img.dtype != np.uint8:
            print(f"❌ dtype={img.dtype}：{p}")
            bad += 1
            continue
        uniq = set(np.unique(img).tolist())
        allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
        if not uniq.issubset(allowed):
            print(f"❌ 含非法 trainId {uniq - allowed}：{p}")
            bad += 1
    if bad == 0:
        print(f"✅ Sanity check 通過：{len(written)} 張 PNG 全部符合規範")
    else:
        print(f"⚠️  {bad}/{len(written)} 張不符規範")


def make_zip(out_root: Path, zip_path: Path) -> None:
    """打包整個 out_root 為 ACDC evaluation server 合規 zip。

    規範要求頂層須有 ``labelTrainIds/`` 目錄,結果檔放其下任意位置即可
    （server 依檔名 ``GOPR..._frame_..*.png`` 在 labelTrainIds/ 下遞迴比對）。
    磁碟上的 out_root 仍保留 <condition>/test/<seq>/... 原始結構,只在打包時
    於每個 arcname 前綴 ``labelTrainIds/``。
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for png in sorted(out_root.rglob('*.png')):
            arcname = Path('labelTrainIds') / png.relative_to(out_root)
            zf.write(png, arcname=str(arcname))
            n += 1
    print(f"📦 已打包：{zip_path}（{zip_path.stat().st_size / 1e6:.1f} MB, "
          f"{n} files, 頂層 labelTrainIds/）")


def parse_args() -> argparse.Namespace:
    repo_root = _SEGANY_ROOT.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--csv', type=str,
                   default=str(repo_root / 'Datasets' / 'acdc_adverse_ref_rgb_test.csv'),
                   help='ACDC test CSV（gt_path 可為空）')
    p.add_argument('--ckpt', type=str,
                   default=str(_SEGANY_ROOT / 'outputs_pair_sam_mask2former_testv15'
                               / 'weather_sam_best_latest.pth'),
                   help='v15 checkpoint 路徑')
    p.add_argument('--out', type=str,
                   default=str(DEFAULT_OUT_DIR),
                   help=f'submission 輸出根目錄（內部會建立 <condition>/test/<seq>/...）。'
                        f'預設絕對路徑：{DEFAULT_OUT_DIR}')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--zip', action='store_true', help='完成後額外打包 submission.zip')
    p.add_argument('--max-samples', type=int, default=None,
                   help='只跑前 N 張（debug 用）')
    p.add_argument('--skip-verify', action='store_true',
                   help='跳過 sanity check（節省時間）')
    p.add_argument('--debug-viz', type=str, default=None,
                   help=f'三聯圖 debug 視覺化的輸出根目錄；不指定時，若有開啟 viz '
                        f'（--viz-per-condition / --viz-all）則自動寫到固定位置：{DEFAULT_VIZ_DIR}')
    p.add_argument('--viz-per-condition', type=int, default=None,
                   help='每個 condition (fog/rain/snow/night) 各抽前 N 張做三聯圖。'
                        '抽樣模式，最常用。')
    p.add_argument('--viz-all', action='store_true',
                   help='對所有樣本輸出三聯圖（小心：2000 張會花很久且佔大量磁碟）。')
    p.add_argument('--viz-only', action='store_true',
                   help='純 debug：只對抽中的樣本跑推論，跳過其他樣本（不產 submission 全集）。'
                        '需與 --viz-per-condition 搭配使用。')
    return p.parse_args()


def _anchor_to_repo_root(p: str | None) -> Path | None:
    """把相對路徑錨定到 repo 根目錄（_SEGANY_ROOT.parent），避免 cwd 影響輸出位置。"""
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (_SEGANY_ROOT.parent / path).resolve()
    return path


def main() -> None:
    args = parse_args()
    out_root = _anchor_to_repo_root(args.out)
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Test CSV    : {args.csv}")
    print(f"Output root : {out_root}")
    print(f"Device      : {args.device}")

    model, _cfg = load_pair_sam_from_ablation(args.ckpt, device=args.device)
    loader = build_test_loader(args.csv, num_workers=args.num_workers)
    csv_df = loader.dataset.data.reset_index(drop=True)

    # viz 開關：--viz-per-condition 或 --viz-all 任一即啟用
    viz_enabled = args.viz_per_condition is not None or args.viz_all
    # 路徑：使用者顯式提供則錨定到 repo root；未提供且 viz 啟用則自動套用固定預設位置
    viz_root = _anchor_to_repo_root(args.debug_viz)
    if viz_root is None and viz_enabled:
        viz_root = DEFAULT_VIZ_DIR
    viz_indices: set[int] | None = None
    if args.viz_per_condition is not None:
        viz_indices = select_stratified_indices(csv_df, args.viz_per_condition)
        print(f"Stratified viz: 每 condition 抽 {args.viz_per_condition} 張，"
              f"總計 {len(viz_indices)} 張")
    if args.viz_only and not viz_enabled:
        raise SystemExit("--viz-only 需搭配 --viz-per-condition N 或 --viz-all 使用")
    if viz_root is not None:
        print(f"Debug viz   : {viz_root}")
    written = dump_predictions(
        model, loader, csv_df, args.device, out_root,
        max_samples=args.max_samples,
        viz_root=viz_root,
        viz_indices=viz_indices,
        viz_only=args.viz_only,
    )
    print(f"✍️  寫出 {len(written)} 張 prediction PNG")

    if not args.skip_verify:
        expected = args.max_samples if args.max_samples is not None else len(csv_df)
        verify_outputs(written, expected_count=expected)

    if args.zip:
        zip_path = out_root.with_suffix('.zip')
        make_zip(out_root, zip_path)


if __name__ == '__main__':
    main()
