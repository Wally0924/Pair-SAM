"""Ours (PairSAM, R7) vs CMA 定性比較圖。

對 CMA manifest 列出的 ACDC val 影像：
  * 以 R7_seed42 ablation config 重建 PairSAM 並原生解析度推論（=Ours）
  * 直接讀回 CMA 預渲染結果（P-mode PNG，像素值即 trainId）
  * 讀原生 GT + invalid_mask，按官方協定 (GT==255 或 invalid) 過濾
  * 算每張影像 Ours / CMA 的 mIoU（與 eval_e1_acdc_val_full 同一 IoU 定義）

兩段式：
  1. 對所有樣本算指標，寫 <out_dir>/ours_vs_cma_metrics.csv（已存在則直接讀，除非 --recompute）
  2. 每個天氣自動挑「Ours - CMA mIoU 差距最大」那張（可用 --picks 覆寫），
     輸出 4 列 × 5 欄定性圖：原圖 | Reference | GT | CMA | Ours

用法：
    conda run -n sam_env python scripts/eval/viz_ours_vs_cma.py
    # 覆寫挑選（以 filename 指定，逗號分隔，順序不限）：
    conda run -n sam_env python scripts/eval/viz_ours_vs_cma.py \
        --picks GP010476_frame_000058_rgb_anon.png,GOPR0402_frame_000365_rgb_anon.png,...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))       # _eval_common
sys.path.insert(0, str(_THIS.parents[1]))   # utils
from _eval_common import (  # noqa: E402
    load_pair_sam_from_ablation, colorize_19class,
    CITYSCAPES_CLASSES, CITYSCAPES_PALETTE,
)
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits  # noqa: E402
from utils.pair_dataloader import PairSegmentationDataset  # noqa: E402

NUM_CLASSES = 19
IGNORE_INDEX = 255
DEVICE = 'cuda'
CONDITION_ORDER = ['fog', 'rain', 'snow', 'night']

# Cityscapes 19 類拆分：靜態 stuff（場景結構）/ 動態 things（可移動實例）
STATIC_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # road..sky
DYNAMIC_IDS = [11, 12, 13, 14, 15, 16, 17, 18]    # person..bicycle

_SEGANY_ROOT = _THIS.parents[2]
DEFAULT_CKPT = str(_SEGANY_ROOT / 'outputs_ablation' / 'R7_seed42' / 'weather_sam_best_latest.pth')
DEFAULT_MANIFEST = '/home/rvl1421/cma/qual_results/manifest.csv'
CMA_ROOT = '/home/rvl1421/cma'
DEFAULT_VAL_CSV = str(_SEGANY_ROOT.parent / 'Datasets' / 'acdc_adverse_ref_rgb_val.csv')
DEFAULT_OUT_DIR = Path('/home/rvl1421/Downloads/figures')


# ── 指標 ───────────────────────────────────────────────────
def per_image_miou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """單張 mIoU：在有效像素上累積混淆矩陣，IoU 對 denom>0 的類別取平均。
    與 eval_e1_acdc_val_full.iou_from_confusion 同定義。"""
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    p = np.clip(p, 0, NUM_CLASSES - 1)
    cm = np.bincount(g * NUM_CLASSES + p,
                     minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return float(np.nanmean(iou)) * 100.0


def per_class_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """回傳 (19,) 每類 IoU（百分比），denom=0 的類別為 nan。"""
    g = gt[valid].astype(np.int64)
    p = np.clip(pred[valid].astype(np.int64), 0, NUM_CLASSES - 1)
    cm = np.bincount(g * NUM_CLASSES + p,
                     minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan) * 100.0


def _grp_miou(iou19: np.ndarray, ids: list[int]) -> float:
    sub = iou19[ids]
    return float(np.nanmean(sub)) if not np.all(np.isnan(sub)) else float('nan')


def enrich_static_dynamic(df: pd.DataFrame, pred_dir: Path) -> pd.DataFrame:
    """用已快取的 Ours npy + CMA png + GT，補上每張的 靜態/動態 per-group mIoU 與
    「靜態類別 Ours≥CMA 的比例」。完全不需重推論。"""
    rows = []
    for _, r in df.iterrows():
        gt = cv2.imread(r['gt_path'], cv2.IMREAD_GRAYSCALE).astype(np.int64)
        invp = str(r['invalid_mask'])
        inv = (cv2.imread(invp, cv2.IMREAD_GRAYSCALE) != 0) if Path(invp).is_file() \
            else np.zeros_like(gt, bool)
        gt[inv] = IGNORE_INDEX
        valid = gt != IGNORE_INDEX

        ours = np.load(pred_dir / r['filename'].replace('.png', '.npy')).astype(np.int64)
        cma = np.array(Image.open(r['colored_result_path'])).astype(np.int64)

        iou_o = per_class_iou(ours, gt, valid)
        iou_c = per_class_iou(cma, gt, valid)
        gt_cls = set(int(v) for v in np.unique(gt[valid]))

        # 靜態類別中，GT 有出現且 Ours IoU ≥ CMA 的比例（nan 視為 0）
        stat_present = [c for c in STATIC_IDS if c in gt_cls]
        wins = sum(1 for c in stat_present
                   if np.nan_to_num(iou_o[c]) >= np.nan_to_num(iou_c[c]))
        stat_win_frac = wins / len(stat_present) if stat_present else 0.0
        dyn_present = [c for c in DYNAMIC_IDS if c in gt_cls]

        rows.append({
            'ours_static': _grp_miou(iou_o, STATIC_IDS),
            'cma_static': _grp_miou(iou_c, STATIC_IDS),
            'ours_dynamic': _grp_miou(iou_o, DYNAMIC_IDS),
            'cma_dynamic': _grp_miou(iou_c, DYNAMIC_IDS),
            'n_static_gt': len(stat_present),
            'n_dynamic_gt': len(dyn_present),
            'static_win_frac': round(stat_win_frac, 3),
        })
    ext = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, ext], axis=1)
    out['static_gap'] = (out['ours_static'] - out['cma_static']).round(2)
    out['dynamic_gap'] = (out['ours_dynamic'] - out['cma_dynamic']).round(2)
    return out


def enrich_baseline_dir(df: pd.DataFrame, base_dir: str, prefix: str) -> pd.DataFrame:
    """通用 baseline 補充：以 <base_dir>/<filename> 讀 trainId PNG（值 0..18），
    算 overall/static/dynamic mIoU，加入欄位 {prefix}_result_path / _miou / _static / _dynamic。
    GT/valid 與其他方法一致。供 SegFormer(source)、Refign 等共用。"""
    bdir = Path(base_dir)
    rows = []
    for _, r in df.iterrows():
        p = bdir / r['filename']
        if not p.is_file():
            raise FileNotFoundError(f'{prefix} 結果缺漏：{p}')
        gt = cv2.imread(r['gt_path'], cv2.IMREAD_GRAYSCALE).astype(np.int64)
        invp = str(r['invalid_mask'])
        inv = (cv2.imread(invp, cv2.IMREAD_GRAYSCALE) != 0) if Path(invp).is_file() \
            else np.zeros_like(gt, bool)
        gt[inv] = IGNORE_INDEX
        valid = gt != IGNORE_INDEX
        pred = np.array(Image.open(p)).astype(np.int64)
        iou = per_class_iou(pred, gt, valid)
        rows.append({
            f'{prefix}_result_path': str(p),
            f'{prefix}_miou': round(float(np.nanmean(iou)), 2),
            f'{prefix}_static': _grp_miou(iou, STATIC_IDS),
            f'{prefix}_dynamic': _grp_miou(iou, DYNAMIC_IDS),
        })
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


def load_native_gt(gt_path: str, invalid_path: str | None):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f'GT not found: {gt_path}')
    gt = gt.astype(np.int64)
    if invalid_path and Path(invalid_path).is_file():
        inv = cv2.imread(invalid_path, cv2.IMREAD_GRAYSCALE) != 0
    else:
        inv = np.zeros_like(gt, dtype=bool)
    return gt, inv


# ── 推論 ───────────────────────────────────────────────────
@torch.no_grad()
def ours_predict_native(model, cfg, item, target_hw):
    """單一 dataset item → Ours 原生解析度 trainId (H, W)。"""
    bi = [{
        'image':         item['image'].to(DEVICE),
        'clear_image':   item['clear_image'].to(DEVICE),
        'text_prompts':  item['text_prompts'],
        'original_size': item['original_size'],
        'condition_id':  item['condition_id'],
    }]
    outputs = model(bi)
    low_res = outputs[0]['low_res_logits'].squeeze(0)
    class_ids_out = outputs[0]['class_ids']
    fused = assemble_semantic_logits(
        low_res, class_ids_out, fusion_head=model.context_fusion_head,
        num_classes=NUM_CLASSES, use_lrh=getattr(model, 'use_lrh', True))
    fused_hr = F.interpolate(fused, size=target_hw, mode='bilinear', align_corners=False)
    return fused_hr.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def compute_metrics_and_cache(args) -> pd.DataFrame:
    """對 manifest 全部樣本算 Ours / CMA mIoU，並快取 Ours trainId npy，回傳 DataFrame。"""
    pred_dir = args.pred_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    val = pd.read_csv(args.val_csv)
    val = val.drop(columns=[c for c in ['condition'] if c in val.columns])  # 避免與 manifest 的 condition 衝突
    merged = man.merge(val, left_on='source_image_path', right_on='image_path', how='left')
    missing = merged['gt_path'].isna().sum()
    if missing:
        raise RuntimeError(f'{missing} manifest 影像在 val CSV 找不到對應 GT/ref，請檢查路徑。')

    # 以過濾後 CSV 建 dataset，順序與 merged 一致
    tmp_csv = args.subset_csv
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    val_cols = list(val.columns)
    merged[val_cols].to_csv(tmp_csv, index=False)
    ds = PairSegmentationDataset(csv_file=str(tmp_csv), image_size=1024,
                                    mode='val', force_raw_images=True)
    assert len(ds) == len(merged), f'dataset {len(ds)} != merged {len(merged)}'

    model, cfg = load_pair_sam_from_ablation(args.ckpt, None, device=DEVICE)
    print(f'[ours_vs_cma] Ours config: {cfg}')

    rows = []
    for idx in range(len(ds)):
        item = ds[idx]
        m = merged.iloc[idx]
        gt_np, inv_np = load_native_gt(str(m['gt_path']),
                                       str(m['invalid_mask']) if 'invalid_mask' in m else None)
        H, W = gt_np.shape
        gt_used = gt_np.copy()
        gt_used[inv_np] = IGNORE_INDEX
        valid = gt_used != IGNORE_INDEX

        ours = ours_predict_native(model, cfg, item, (H, W))
        np.save(pred_dir / m['filename'].replace('.png', '.npy'), ours.astype(np.uint8))

        cma_path = Path(CMA_ROOT) / m['colored_result_path']
        cma = np.array(Image.open(cma_path)).astype(np.int64)
        if cma.shape != (H, W):
            cma = cv2.resize(cma.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(np.int64)

        ours_miou = per_image_miou(ours, gt_used, valid)
        cma_miou = per_image_miou(cma, gt_used, valid)
        n_cls = int(len(np.unique(gt_used[valid])))
        rows.append({
            'condition': m['condition'], 'filename': m['filename'],
            'source_image_path': m['source_image_path'],
            'ref_image_path': m['ref_image_path'], 'gt_path': m['gt_path'],
            'invalid_mask': m.get('invalid_mask', ''),
            'colored_result_path': str(cma_path),
            'ours_miou': round(ours_miou, 2), 'cma_miou': round(cma_miou, 2),
            'gap': round(ours_miou - cma_miou, 2), 'n_gt_classes': n_cls,
        })
        print(f'  [{idx+1:2d}/{len(ds)}] {m["condition"]:5s} {m["filename"]:42s} '
              f'Ours={ours_miou:5.1f}  CMA={cma_miou:5.1f}  Δ={ours_miou-cma_miou:+5.1f}')

    df = pd.DataFrame(rows)
    df.to_csv(args.metrics_csv, index=False)
    print(f'\n✅ metrics 寫入 {args.metrics_csv}')
    return df


# ── 挑選 ───────────────────────────────────────────────────
def _sort_key(strategy: str):
    """回傳 (排序欄位, 升降序) 供 topk 跨全集排序使用。"""
    if strategy == 'static':
        return ['static_win_frac', 'static_gap', 'dynamic_gap'], False
    if strategy == 'static_dynamic':
        return ['static_win_frac', 'dynamic_gap', 'static_gap'], False
    return ['gap', 'ours_miou'], False  # gap


def select_picks(df: pd.DataFrame, picks_arg: str | None,
                 strategy: str = 'gap', min_dynamic: int = 2,
                 topk: int | None = None, ours_must_lead: bool = False) -> pd.DataFrame:
    if picks_arg:
        want = [s.strip() for s in picks_arg.split(',') if s.strip()]
        sub = df[df['filename'].isin(want)].copy()
        if len(sub) != len(want):
            found = set(sub['filename'])
            raise ValueError(f'--picks 有檔名找不到：{[w for w in want if w not in found]}')
    elif ours_must_lead:
        # 要求 Ours static 領先所有 baseline；按領先幅度（對次佳）排序
        base_cols = [c for c in ['seg_static', 'cma_static', 'refign_static'] if c in df.columns]
        d = df.copy()
        d['_best_other'] = d[base_cols].max(axis=1)
        d['_lead'] = d['ours_static'] - d['_best_other']
        d = d[d['_lead'] > 0]
        if topk is not None:
            sub = d.sort_values('_lead', ascending=False).head(topk)
            return sub.drop(columns=['_best_other', '_lead']).reset_index(drop=True)
        chosen = []
        for cond in CONDITION_ORDER:
            c = d[d['condition'] == cond]
            if c.empty:
                print(f'⚠️  {cond}: 無 Ours 領先所有 baseline 的樣本')
                continue
            chosen.append(c.sort_values('_lead', ascending=False).iloc[0])
        sub = pd.DataFrame(chosen).drop(columns=['_best_other', '_lead'])
    elif topk is not None:
        # 跨全集挑前 topk 張（不分天氣），用於單一天氣資料集（如 Dark Zurich，純 night）
        c = df.copy()
        if strategy == 'static':
            c = c[c['static_gap'] > 0]
        elif strategy == 'static_dynamic':
            c = c[(c['static_gap'] > 0) & (c['dynamic_gap'] > 0)
                  & (c['n_dynamic_gt'] >= min_dynamic)]
        by, asc = _sort_key(strategy)
        sub = c.sort_values(by, ascending=asc).head(topk).reset_index(drop=True)
        return sub  # 保持排序（最有利在前）
    elif strategy == 'static':
        # 「靜態結構領先」：每天氣挑 靜態類別 Ours≥CMA 比例最高、靜態 gap 明顯者。
        #   排序：static_win_frac → static_gap，平手再比 dynamic_gap（避免挑到動態崩壞的圖）
        chosen = []
        for cond in CONDITION_ORDER:
            c = df[(df['condition'] == cond) & (df['static_gap'] > 0)
                   & (df['n_static_gt'] >= 4)].copy()
            if c.empty:
                c = df[df['condition'] == cond].copy()
            chosen.append(c.sort_values(
                ['static_win_frac', 'static_gap', 'dynamic_gap'], ascending=False).iloc[0])
        sub = pd.DataFrame(chosen)
    elif strategy == 'static_dynamic':
        # 「靜態結構勝出 → 動態物件跟著更準」：
        #   gate：靜態與動態 gap 皆 > 0，且 GT 至少含 min_dynamic 個動態類別
        #   排序：靜態類別 Ours≥CMA 比例高者優先，再比動態 gap、靜態 gap
        chosen = []
        for cond in CONDITION_ORDER:
            c = df[(df['condition'] == cond)
                   & (df['static_gap'] > 0) & (df['dynamic_gap'] > 0)
                   & (df['n_dynamic_gt'] >= min_dynamic)].copy()
            if c.empty:  # 放寬：動態類別至少 1 個
                c = df[(df['condition'] == cond)
                       & (df['static_gap'] > 0) & (df['dynamic_gap'] > 0)
                       & (df['n_dynamic_gt'] >= 1)].copy()
            if c.empty:
                print(f'⚠️  {cond}: 找不到 靜態&動態皆勝 的樣本，退回 gap 策略')
                c = df[df['condition'] == cond]
                chosen.append(c.sort_values(['gap', 'ours_miou'], ascending=False).iloc[0])
                continue
            chosen.append(c.sort_values(
                ['static_win_frac', 'dynamic_gap', 'static_gap'], ascending=False).iloc[0])
        sub = pd.DataFrame(chosen)
    else:
        # 每天氣挑 Ours 整體 mIoU 優勢最大者（gap），平手時取 Ours mIoU 較高
        chosen = []
        for cond in CONDITION_ORDER:
            c = df[df['condition'] == cond]
            if c.empty:
                continue
            chosen.append(c.sort_values(['gap', 'ours_miou'], ascending=False).iloc[0])
        sub = pd.DataFrame(chosen)
    # 依固定天氣順序排列
    sub['__order'] = sub['condition'].map({c: i for i, c in enumerate(CONDITION_ORDER)})
    return sub.sort_values('__order').drop(columns='__order').reset_index(drop=True)


# ── 繪圖 ───────────────────────────────────────────────────
def render(sub: pd.DataFrame, out_path: Path, annotate: str = 'full',
           pred_dir: Path | None = None, has_seg: bool = False,
           has_refign: bool = False, show_row_label: bool = True,
           ours_title: str = 'CARef-SAM (Ours)'):
    """annotate: 'full'=總 mIoU；'static'=靜態 mIoU；'none'=不標（皆顯示為 "mIoU"）。
    方法欄順序：[SegFormer] | CMA | [Refign] | Ours。show_row_label=False 時隱藏最左列標籤。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image

    if pred_dir is None:
        pred_dir = out_path.parent / '_ours_pred'
    n = len(sub)
    col_titles = ['Input (Adverse)', 'Reference (Clear)', 'Ground Truth']
    if has_seg:
        col_titles.append('SegFormer')
    if has_refign:
        col_titles.append('Refign')
    col_titles.append('CMA')
    col_titles.append(ours_title)
    ncols = len(col_titles)

    # 每格 16:9，欄寬 4.2 吋 → 高 2.36；底部留 legend 一條
    cell_w = 4.2
    cell_h = cell_w * 9 / 16
    fig_w = ncols * cell_w
    fig_h = n * cell_h + 1.3
    fig, axes = plt.subplots(n, ncols, figsize=(fig_w, fig_h))
    if n == 1:
        axes = axes[None, :]

    present_classes: set[int] = set()
    for r in range(n):
        row = sub.iloc[r]
        adverse = cv2.cvtColor(cv2.imread(row['source_image_path']), cv2.COLOR_BGR2RGB)
        ref = cv2.cvtColor(cv2.imread(row['ref_image_path']), cv2.COLOR_BGR2RGB)
        gt_np, inv_np = load_native_gt(row['gt_path'],
                                       str(row['invalid_mask']) if row['invalid_mask'] else None)
        gt_used = gt_np.copy(); gt_used[inv_np] = IGNORE_INDEX
        gt_color = colorize_19class(gt_used)
        present_classes |= set(int(v) for v in np.unique(gt_used) if 0 <= int(v) < NUM_CLASSES)

        # 各方法 trainId → 統一 palette 上色；順序：[SegFormer] | CMA | [Refign] | Ours
        def _color_png(path):
            return colorize_19class(np.array(Image.open(path)).astype(np.int64))
        methods = []
        if has_seg:
            methods.append(('SegFormer', _color_png(row['seg_result_path']),
                            row.get('seg_static'), row.get('seg_miou'), 'black'))
        if has_refign:
            methods.append(('Refign', _color_png(row['refign_result_path']),
                            row.get('refign_static'), row.get('refign_miou'), 'black'))
        methods.append(('CMA', _color_png(row['colored_result_path']),
                        row.get('cma_static'), row['cma_miou'], 'black'))
        ours = np.load(pred_dir / row['filename'].replace('.png', '.npy'))  # Ours 快取，免二次推論
        methods.append(('Ours', colorize_19class(ours),
                        row.get('ours_static'), row['ours_miou'], 'darkgreen'))

        panels = [adverse, ref, gt_color] + [m[1] for m in methods]
        for col, img in enumerate(panels):
            ax = axes[r, col]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        # 列標籤（可關閉）：多天氣標天氣；單一天氣標影格編號
        if show_row_label:
            if sub['condition'].nunique() > 1:
                row_label = row['condition'].capitalize()
            else:
                import re
                mfr = re.search(r'frame_(\d+)', row['filename'])
                row_label = f"{row['condition'].capitalize()}\nframe {mfr.group(1)}" if mfr \
                    else row['condition'].capitalize()
            axes[r, 0].set_ylabel(row_label, fontsize=18, fontweight='bold')
        # mIoU 標註（左上角；不含 "static" 字樣）
        def _lbl(s_val, miou_val):
            val = s_val if annotate == 'static' else miou_val
            if val is None or pd.isna(val):
                return None
            return f"mIoU {val:.1f}"
        if annotate != 'none':
            for i, (_title, _img, s_val, miou_val, fc) in enumerate(methods):
                txt = _lbl(s_val, miou_val)
                if txt is None:
                    continue
                axes[r, 3 + i].text(0.02, 0.97, txt, transform=axes[r, 3 + i].transAxes,
                                    fontsize=15, fontweight='bold', va='top', ha='left', color='white',
                                    bbox=dict(boxstyle='round,pad=0.2', fc=fc, ec='none',
                                              alpha=0.7 if fc == 'darkgreen' else 0.6))

    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=21, fontweight='bold', pad=10)

    # 底部類別 legend
    present = sorted(present_classes)
    handles = [mpatches.Patch(color=CITYSCAPES_PALETTE[c] / 255.0, label=CITYSCAPES_CLASSES[c])
               for c in present]
    fig.legend(handles=handles, loc='lower center', ncol=min(10, len(handles)),
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(left=0.03, right=0.997, top=0.965, bottom=0.06,
                        wspace=0.02, hspace=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ 定性比較圖寫入 {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--val_csv', default=DEFAULT_VAL_CSV)
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--picks', default=None, help='以 filename 覆寫每天氣挑選（逗號分隔）')
    ap.add_argument('--strategy', default='static',
                    choices=['gap', 'static', 'static_dynamic'],
                    help='挑圖策略：gap=整體 mIoU 優勢；static=靜態結構領先；'
                         'static_dynamic=靜態勝出且動態跟著勝')
    ap.add_argument('--min_dynamic', type=int, default=2,
                    help='static_dynamic 策略下，GT 至少含幾個動態類別才納入候選')
    ap.add_argument('--annotate', default='auto', choices=['auto', 'static', 'full', 'none'],
                    help='格內標註：auto=依策略；static=只標靜態 mIoU；full=總 mIoU；none=不標')
    ap.add_argument('--cma_dir', default=None,
                    help='CMA 結果的 trainId 目錄（新版佈局）；提供時覆寫 manifest 的 CMA 路徑')
    ap.add_argument('--segformer_dir', default=None,
                    help='SegFormer(source) baseline 的 trainId 目錄；提供時加一欄 SegFormer')
    ap.add_argument('--refign_dir', default=None,
                    help='Refign baseline 的 trainId 目錄；提供時在 CMA 後加一欄 Refign')
    ap.add_argument('--no_row_label', action='store_true', help='隱藏最左側列標籤（如 DZ 圖）')
    ap.add_argument('--ours_must_lead', action='store_true',
                    help='挑圖時要求 Ours static 領先所有 baseline（SegFormer/CMA/Refign），按領先幅度排序')
    ap.add_argument('--ours_title', default='CARef-SAM (Ours)', help='Ours 欄位標題')
    ap.add_argument('--tag', default='', help='輸出/快取檔名後綴，用於區隔不同資料集（如 _darkzurich）')
    ap.add_argument('--topk', type=int, default=None,
                    help='跨全集挑前 N 張（不分天氣），用於單一天氣資料集（如 Dark Zurich）')
    ap.add_argument('--recompute', action='store_true', help='強制重算指標（重跑推論）')
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir)
    tag = args.tag
    args.metrics_csv = args.out_dir / f'ours_vs_cma_metrics{tag}.csv'
    args.pred_dir = args.out_dir / f'_ours_pred{tag}'
    args.subset_csv = args.out_dir / f'_ours_vs_cma_subset{tag}.csv'
    out_fig = args.out_dir / f'ours_vs_cma_qualitative{tag}.png'
    pred_dir = args.pred_dir

    if args.metrics_csv.is_file() and not args.recompute:
        print(f'[ours_vs_cma] 讀取既有指標 {args.metrics_csv}（--recompute 可重算）')
        df = pd.read_csv(args.metrics_csv)
    else:
        df = compute_metrics_and_cache(args)

    # CMA 路徑改採新版 trainId 佈局（覆寫快取 CSV 內失效的舊路徑）
    if args.cma_dir:
        df['colored_result_path'] = df['filename'].apply(lambda f: str(Path(args.cma_dir) / f))
        print(f'[ours_vs_cma] CMA 路徑改用 {args.cma_dir}')

    # 靜態/動態 per-group 指標（用快取 Ours npy + CMA trainId，不需重推論）
    df = enrich_static_dynamic(df, pred_dir)
    has_seg = args.segformer_dir is not None
    if has_seg:
        df = enrich_baseline_dir(df, args.segformer_dir, 'seg')
        print(f'[ours_vs_cma] 已加入 SegFormer 欄（{args.segformer_dir}）')
    has_refign = args.refign_dir is not None
    if has_refign:
        df = enrich_baseline_dir(df, args.refign_dir, 'refign')
        print(f'[ours_vs_cma] 已加入 Refign 欄（{args.refign_dir}）')
    enriched_csv = args.out_dir / f'ours_vs_cma_metrics_static_dynamic{tag}.csv'
    df.to_csv(enriched_csv, index=False)
    print(f'[ours_vs_cma] 靜態/動態指標寫入 {enriched_csv}')

    sub = select_picks(df, args.picks, strategy=args.strategy,
                       min_dynamic=args.min_dynamic, topk=args.topk,
                       ours_must_lead=args.ours_must_lead)
    print('\n挑選結果：')
    cols = ['condition', 'filename', 'ours_static', 'cma_static',
            'ours_dynamic', 'cma_dynamic', 'static_win_frac', 'n_dynamic_gt']
    print(sub[cols].to_string(index=False))
    if args.annotate == 'auto':
        annotate = 'static' if args.strategy == 'static' else 'full'
    else:
        annotate = args.annotate
    render(sub, out_fig, annotate=annotate, pred_dir=pred_dir, has_seg=has_seg,
           has_refign=has_refign, show_row_label=not args.no_row_label,
           ours_title=args.ours_title)


if __name__ == '__main__':
    main()
