"""§4.8.1 置信度分析：以 R7_seed42 在 ACDC val（406 張）計算
  (a) 置信度 vs mIoU 去相關散點圖（fig:weather_results，含 Pearson r）
  (b) 靜態區 vs 動態區平均置信度（§4.8.1 [0.X] 門檻）
並產出 fog/night 置信度圖（fig:confidence_maps）。純推論，僅跑 fusion_module.pre_align。
"""
import sys, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from _eval_common import (load_weather_sam_model, denorm_image,
                          pick_first_per_condition, CONDITION_NAMES, DEFAULT_VAL_CSV)
sys.path.insert(0, str(_THIS.parents[2]))
from utils.weather_dataloader import WeatherSegmentationDataset

DEVICE = 'cuda'
R7_CKPT = str(_THIS.parents[2] / 'outputs_ablation' / 'R7_seed42' / 'weather_sam_best_latest.pth')
FIG_OUT = '/home/rvl1421/Downloads/figures'
# trainId 分組：靜態 stuff vs 動態 thing
STATIC = set(range(0, 11))   # road..sky
DYNAMIC = set(range(11, 19)) # person..bicycle
os.makedirs(FIG_OUT, exist_ok=True)


def conf_1024(model, adverse, clear):
    """回傳 (1024,1024) 的置信度 map（0~1）。"""
    model.fusion_module.pre_align(adverse, clear)
    conf_lr = model.fusion_module._last_confidence_map.to(DEVICE)   # (1,1,64,64)
    conf_hi = F.interpolate(conf_lr, size=(1024, 1024), mode='bilinear', align_corners=False)
    return conf_hi.clamp(0, 1).squeeze().cpu().numpy()


def main():
    model = load_weather_sam_model(R7_CKPT, device=DEVICE)
    ds = WeatherSegmentationDataset(csv_file=DEFAULT_VAL_CSV, image_size=1024,
                                    mode='val', force_raw_images=True)
    n = len(ds)
    print(f'val samples: {n}  ckpt: R7_seed42')

    cond_sum = {c: [0.0, 0] for c in CONDITION_NAMES.values()}   # [conf_sum, pix_cnt]
    static_sum = [0.0, 0]; dynamic_sum = [0.0, 0]

    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            adverse = item['image'].unsqueeze(0).to(DEVICE)
            clear = item['clear_image'].unsqueeze(0).to(DEVICE)
            conf = conf_1024(model, adverse, clear)              # (1024,1024)
            gt = item['gt_mask'].numpy()                          # (1024,1024) trainId
            inv = item.get('invalid_mask')
            valid = (gt != 255)
            if inv is not None:
                valid &= (~inv.numpy())
            cond = CONDITION_NAMES[int(item['condition_id'])]
            # 逐條件（全有效像素）
            cond_sum[cond][0] += float(conf[valid].sum()); cond_sum[cond][1] += int(valid.sum())
            # 靜態/動態區
            sm = valid & np.isin(gt, list(STATIC))
            dm = valid & np.isin(gt, list(DYNAMIC))
            static_sum[0] += float(conf[sm].sum()); static_sum[1] += int(sm.sum())
            dynamic_sum[0] += float(conf[dm].sum()); dynamic_sum[1] += int(dm.sum())
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{n}')

    stats = {
        'per_condition_conf': {c: round(s[0]/s[1], 4) for c, s in cond_sum.items()},
        'static_region_conf': round(static_sum[0]/static_sum[1], 4),
        'dynamic_region_conf': round(dynamic_sum[0]/dynamic_sum[1], 4),
    }
    print('STATS:', json.dumps(stats, ensure_ascii=False))
    json.dump(stats, open(f'{FIG_OUT}/ch4_confidence_stats.json', 'w'), ensure_ascii=False, indent=2)

    # ---- fig:weather_results（選項 A）：置信度 vs mIoU 去相關散點圖 ----
    # x=逐條件 mIoU（取自 R7_seed42 e1_results），y=逐條件平均置信度，標 Pearson r
    conds = ['fog', 'rain', 'snow', 'night']
    e1 = json.load(open(_THIS.parents[2] / 'outputs_ablation' / 'R7_seed42' / 'e1_results.json'))
    miou = [e1['per_condition_miou'][c] * 100 for c in conds]
    conf = [stats['per_condition_conf'][c] for c in conds]
    r = float(np.corrcoef(miou, conf)[0, 1])
    colors = ['#4C9F70', '#4C72B0', '#9C7CB8', '#C44E52']
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(miou, conf, c=colors, s=180, edgecolor='black', linewidth=0.8, zorder=3)
    for x, y, c in zip(miou, conf, conds):
        ax.annotate(c.capitalize(), (x, y), textcoords='offset points', xytext=(8, 6),
                    fontsize=12, fontweight='bold')
    a, b = np.polyfit(miou, conf, 1)                              # 最小平方擬合線：視覺化「幾乎無趨勢」
    xs = np.array([min(miou) - 2, max(miou) + 2])
    ax.plot(xs, a * xs + b, ls='--', color='gray', lw=1.3, zorder=2)
    ax.set_xlabel('Per-condition mIoU (%)', fontsize=12)
    ax.set_ylabel('Mean UAWarpC confidence', fontsize=12)
    ax.set_title(f'Confidence vs mIoU (ACDC val), Pearson r = {r:.2f}', fontsize=12)
    ax.grid(ls=':', alpha=0.5)
    plt.tight_layout(); plt.savefig(f'{FIG_OUT}/ch4_conf_by_condition.png', dpi=200); plt.close()
    stats['conf_miou_pearson_r'] = round(r, 4)                    # 把 r 一併寫回 stats
    json.dump(stats, open(f'{FIG_OUT}/ch4_confidence_stats.json', 'w'), ensure_ascii=False, indent=2)
    print(f'saved ch4_conf_by_condition.png (scatter), Pearson r = {r:.3f}')

    # ---- fig:confidence_maps：fog vs night 置信度熱圖 ----
    picked = pick_first_per_condition(DEFAULT_VAL_CSV)
    sel = {0: picked[0], 3: picked[3]}  # fog, night
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    col_titles = ['Adverse (Target)', 'Warped Ref × Confidence', 'Confidence map']
    with torch.no_grad():
        for row, cid in enumerate([0, 3]):
            item = ds[sel[cid]]
            adverse = item['image'].unsqueeze(0).to(DEVICE)
            clear = item['clear_image'].unsqueeze(0).to(DEVICE)
            conf = conf_1024(model, adverse, clear)
            adv_np = denorm_image(adverse[0])
            warped_np = denorm_image(clear[0]).astype(np.float32)
            white = np.ones_like(warped_np) * 255.0
            wc = (warped_np * conf[..., None] + white * (1 - conf[..., None])).astype(np.uint8)
            axes[row, 0].imshow(adv_np)
            axes[row, 1].imshow(wc)
            im = axes[row, 2].imshow(conf, cmap='viridis', vmin=0, vmax=1)
            axes[row, 0].set_ylabel(CONDITION_NAMES[cid].capitalize(), fontsize=22, fontweight='bold')
            for col in range(3):
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        for col, t in enumerate(col_titles):
            axes[0, col].set_title(t, fontsize=16, fontweight='bold')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    plt.savefig(f'{FIG_OUT}/ch4_confidence_maps.png', dpi=150, bbox_inches='tight'); plt.close()
    print('saved ch4_confidence_maps.png')


if __name__ == '__main__':
    main()
