"""
diagnose_location_attn.py

診斷 location token 在 Mask Decoder Transformer 中的 attention 利用程度。

Token 序列結構 (per class k):
  [iou(0), mask0(1), mask1(2), mask2(3), mask3(4), text(5), location(6)]

觀察指標:
  1. cross_attn_token_to_image: location token (query idx=6) 對 image 的 attention 分布
     → 若 entropy 接近 log(4096)（均勻分布），代表模型無法從 GPS 找到有意義的 image 區域
     → 若 entropy 明顯低於均勻值，代表 GPS 有引導視覺注意力

  2. self_attn (token-to-token): 其他 token 對 location token (key idx=6) 的 attention 強度
     → 若 text token (idx=5) 幾乎不看 location token，代表兩者缺乏交互

使用方式:
  python diagnose_location_attn.py
"""

import torch
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from segment_anything.build_weather_sam import build_weather_sam_vit_h
from utils.weather_dataloader import WeatherSegmentationDataset
from segment_anything.weather_predictor import WeatherSamPredictor
from segment_anything.modeling.transformer import Attention

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
CHECKPOINT_PATH = "/home/rvl1421/SAM_research-1/segment-anything/outputs_weather_sam_all_data_testv16/weather_sam_best_latest.pth"
TEST_CSV_PATH   = "/home/rvl1421/SAM_research-1/Datasets/test_with_gps.csv"
OUTPUT_DIR      = "diagnose_location_attn_results"
NUM_SAMPLES     = 10   # 跑幾張圖就夠診斷，不需要全跑
LOCATION_IDX    = 6    # token sequence 中 location token 的固定位置
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# Monkey-patch: 攔截 Attention.forward
# ─────────────────────────────────────────
_attn_store: dict = {}   # { module_name: List[attn_tensor] }

_original_forward = Attention.forward

def _patched_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_ = self.q_proj(q)
    k_ = self.k_proj(k)
    v_ = self.v_proj(v)

    q_ = self._separate_heads(q_, self.num_heads)
    k_ = self._separate_heads(k_, self.num_heads)
    v_ = self._separate_heads(v_, self.num_heads)

    _, _, _, c_per_head = q_.shape
    attn = q_ @ k_.permute(0, 1, 3, 2)          # (B, heads, Q, K)
    attn = torch.softmax(attn / math.sqrt(c_per_head), dim=-1)

    # 存入全域容器
    name = getattr(self, '_diag_name', 'unknown')
    if name not in _attn_store:
        _attn_store[name] = []
    _attn_store[name].append(attn.detach().cpu())

    out = attn @ v_
    out = self._recombine_heads(out)
    return self.out_proj(out)


def install_patch(model):
    """為每個 Attention module 命名並替換 forward。"""
    Attention.forward = _patched_forward
    transformer = model.mask_decoder.transformer
    for layer_idx, layer in enumerate(transformer.layers):
        layer.self_attn._diag_name             = f"L{layer_idx}_self"
        layer.cross_attn_token_to_image._diag_name = f"L{layer_idx}_tok2img"
        layer.cross_attn_image_to_token._diag_name = f"L{layer_idx}_img2tok"
    # final attention after all layers
    if hasattr(transformer, 'final_attn_token_to_image'):
        transformer.final_attn_token_to_image._diag_name = "final_tok2img"


def remove_patch():
    Attention.forward = _original_forward


# ─────────────────────────────────────────
# 分析工具
# ─────────────────────────────────────────
def entropy(attn_row: np.ndarray) -> float:
    """計算 attention 分布的 Shannon entropy（bits）。"""
    p = attn_row + 1e-9
    return float(-np.sum(p * np.log2(p)))


def uniform_entropy(n: int) -> float:
    return math.log2(n)


def analyze_sample(sample_idx: int):
    """
    對單一 sample 的 _attn_store 做分析，回傳摘要 dict。
    """
    results = {}

    for name, attn_list in _attn_store.items():
        # 每次 forward 可能累積多個呼叫 (K prompts)，取 mean
        attn = torch.stack(attn_list, dim=0).mean(dim=0)  # (B, heads, Q, K_seq)
        attn = attn.mean(dim=0)   # avg over batch  → (heads, Q, K_seq)
        attn_np = attn.numpy()

        n_heads, n_q, n_k = attn_np.shape

        if "tok2img" in name or "final_tok2img" in name:
            # cross: Q = token seq (≥7), K = image patches
            if n_q > LOCATION_IDX:
                loc_row = attn_np[:, LOCATION_IDX, :]          # (heads, 4096)
                text_row = attn_np[:, LOCATION_IDX - 1, :]    # (heads, 4096)  (text token)
                loc_ent  = np.mean([entropy(loc_row[h])  for h in range(n_heads)])
                text_ent = np.mean([entropy(text_row[h]) for h in range(n_heads)])
                max_ent  = uniform_entropy(n_k)
                results[name] = {
                    "type": "tok2img",
                    "loc_entropy":  loc_ent,
                    "text_entropy": text_ent,
                    "uniform_entropy": max_ent,
                    "loc_focus_ratio":  1.0 - loc_ent  / max_ent,   # 越高代表越集中
                    "text_focus_ratio": 1.0 - text_ent / max_ent,
                }

        elif "self" in name:
            # self-attn: Q = K = token seq
            if n_q > LOCATION_IDX and n_k > LOCATION_IDX:
                # 其他 token 對 location token 的 attention（column）
                col = attn_np[:, :, LOCATION_IDX]               # (heads, Q)
                # text token 本身對 location 的注意力
                text_to_loc = col[:, LOCATION_IDX - 1].mean()   # scalar
                avg_col = col.mean()                              # 平均關注度
                results[name] = {
                    "type": "self",
                    "avg_attn_to_loc":  float(avg_col),
                    "text_to_loc_attn": float(text_to_loc),
                    # 基準: 若均勻則每格 = 1/n_q
                    "uniform_baseline": 1.0 / n_q,
                }

    return results


# ─────────────────────────────────────────
# 視覺化
# ─────────────────────────────────────────
def plot_summary(all_results: list, save_path: str):
    """
    all_results: List[dict]，每個 sample 的 analyze_sample 結果。
    繪製跨 sample 的統計圖。
    """
    # 只挑 tok2img 層做圖（最關鍵）
    tok2img_keys = [k for k in all_results[0] if "tok2img" in k]

    n_layers = len(tok2img_keys)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for ax, key in zip(axes, tok2img_keys):
        loc_foci  = [r[key]["loc_focus_ratio"]  for r in all_results if key in r]
        text_foci = [r[key]["text_focus_ratio"] for r in all_results if key in r]

        x = np.arange(len(loc_foci))
        ax.bar(x - 0.2, loc_foci,  0.4, label="Location token", color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, text_foci, 0.4, label="Text token",     color="coral",     alpha=0.8)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, label="Uniform (baseline)")
        ax.set_title(key)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Focus Ratio (1 - entropy/max_entropy)\nhigher = more focused")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.suptitle("Location vs Text Token Attention Focus (tok→img cross-attention)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot saved] {save_path}")


def print_summary(all_results: list):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    keys = list(all_results[0].keys())

    for key in keys:
        vals = [r[key] for r in all_results if key in r]
        t = vals[0]["type"]
        print(f"\n[{key}]")

        if t == "tok2img":
            loc_foci  = np.mean([v["loc_focus_ratio"]  for v in vals])
            text_foci = np.mean([v["text_focus_ratio"] for v in vals])
            loc_ent   = np.mean([v["loc_entropy"]      for v in vals])
            text_ent  = np.mean([v["text_entropy"]     for v in vals])
            max_ent   = vals[0]["uniform_entropy"]
            print(f"  [Token→Image cross-attention]")
            print(f"  Uniform entropy (baseline) : {max_ent:.3f} bits")
            print(f"  Location token entropy     : {loc_ent:.3f} bits  | focus ratio: {loc_foci:.3f}")
            print(f"  Text    token entropy      : {text_ent:.3f} bits  | focus ratio: {text_foci:.3f}")
            if loc_foci < 0.05:
                verdict = "⚠  NEAR-UNIFORM — location token 幾乎沒有聚焦，對 image 查詢無效"
            elif loc_foci < text_foci * 0.5:
                verdict = "△  WEAK — location token 有一定聚焦，但明顯弱於 text token"
            else:
                verdict = "✓  ACTIVE — location token 有效聚焦，與 text token 同量級"
            print(f"  verdict: {verdict}")

        elif t == "self":
            avg   = np.mean([v["avg_attn_to_loc"]  for v in vals])
            t2l   = np.mean([v["text_to_loc_attn"] for v in vals])
            unif  = vals[0]["uniform_baseline"]
            print(f"  [Token self-attention → location token column]")
            print(f"  Uniform baseline (1/n_tokens) : {unif:.4f}")
            print(f"  Avg attention to location     : {avg:.4f}  ({avg/unif:.2f}x uniform)")
            print(f"  Text-token → location attn    : {t2l:.4f}  ({t2l/unif:.2f}x uniform)")
            if avg < unif * 0.5:
                verdict = "⚠  IGNORED — 其他 token 幾乎不關注 location token"
            elif avg > unif * 1.5:
                verdict = "✓  ATTENDED — location token 在自注意力中被積極利用"
            else:
                verdict = "△  MARGINAL — 接近均勻，輕微利用"
            print(f"  verdict: {verdict}")


# ─────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────
def main():
    print(f"Loading model from {CHECKPOINT_PATH} ...")
    model = build_weather_sam_vit_h(checkpoint=CHECKPOINT_PATH)
    model.to(DEVICE)
    model.eval()

    install_patch(model)

    predictor = WeatherSamPredictor(model)

    test_ds = WeatherSegmentationDataset(csv_file=TEST_CSV_PATH, mode='val')
    test_ds.has_cached_features = False
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=WeatherSegmentationDataset.collate_fn
    )

    classes = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation",
        "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    all_results = []

    with torch.no_grad():
        for sample_idx, batch in enumerate(test_loader):
            if sample_idx >= NUM_SAMPLES:
                break

            sample = {
                'reference_mask': batch['reference_mask'][0],
                'ref_void_mask':  batch['ref_void_mask'][0],
                'location':       batch['location'][0],
                'original_size':  batch['original_size'][0],
            }
            if 'image' in batch:
                sample['image'] = batch['image'][0]
            if 'image_embedding' in batch:
                sample['image_embedding'] = batch['image_embedding'][0]

            gt_mask_np = batch['gt_mask'][0].numpy()
            active_prompts = []
            for cls_id in np.unique(gt_mask_np):
                if cls_id < len(classes):
                    active_prompts.append(classes[cls_id])
            if not active_prompts:
                active_prompts = ["road"]

            # 清空上一個 sample 的紀錄
            _attn_store.clear()

            # 跑 inference（此時 hook 會自動填入 _attn_store）
            predictor.set_image_data(
                image=sample.get('image', None).to(DEVICE) if 'image' in sample else None,
                image_embedding=sample.get('image_embedding', None).to(DEVICE) if 'image_embedding' in sample else None,
                reference_mask=sample['reference_mask'].to(DEVICE),
                ref_void_mask=sample['ref_void_mask'].to(DEVICE),
                original_size=sample['original_size'],
            )
            predictor.predict(
                text_prompts=active_prompts,
                location=sample['location'].to(DEVICE),
                multimask_output=True,
            )

            result = analyze_sample(sample_idx)
            all_results.append(result)
            print(f"[Sample {sample_idx:02d}] GPS={batch['location'][0].numpy()}  prompts={active_prompts[:3]}...")

    remove_patch()

    if all_results:
        print_summary(all_results)
        plot_summary(all_results, save_path=os.path.join(OUTPUT_DIR, "attn_focus_summary.png"))

    print(f"\nDone. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
