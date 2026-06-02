# Rare Class Sampling (RCS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 DAFormer 風格的 Rare Class Sampling（先依 P(c) 抽類別、再抽含該類影像）導入 WeatherSAM 監督訓練，並把 RCS 納入第 4.9 節消融框架（R8=新 FULL）。

**Architecture:** RCS 是純訓練期資料取樣機制，不改模型 → eval / 模型建構不受影響。一次性 precompute 每影像類別表 → `RareClassSampler`（torch Sampler，seed 綁定可重現）→ train.py 在 `--rcs` 時以 sampler 取代 shuffle。f(c) 由 1600-train 實際像素計數重算；`--rcs` 預設 on。

**Tech Stack:** PyTorch `Sampler`、numpy、cv2、pytest（`conda run -n sam_env`，但本專案指令已改直接 `python`；測試用 `python -m pytest`）。

> 設計依據：[`docs/superpowers/specs/2026-06-03-rcs-design.md`](../specs/2026-06-03-rcs-design.md)。連動修訂 [`2026-06-01-ablation-experiment-design.md`](../specs/2026-06-01-ablation-experiment-design.md)、[`2026-06-01-paper-rewrite-4.9-ablation.md`](../specs/2026-06-01-paper-rewrite-4.9-ablation.md)。

---

## 檔案結構

**新增：**
- `segment-anything/scripts/precompute_class_presence.py` — 掃 train GT → 每影像類別表 + 全域像素計數 → `class_presence.json`
- `segment-anything/utils/rare_class_sampler.py` — `RareClassSampler(torch.utils.data.Sampler)`
- `segment-anything/tests/test_precompute_class_presence.py`、`segment-anything/tests/test_rare_class_sampler.py`

**修改：**
- `segment-anything/train.py` — `--rcs`/`--rcs_temp`/`--class_presence` flags、建 sampler、config.json
- `segment-anything/scripts/aggregate_ablation.py` — 累積表 R1–R8、C2 改用自身 run、FULL=R8
- `segment-anything/run_ablation.sh`、`segment-anything/ABLATION_RUNBOOK.md`、`docs/superpowers/specs/2026-06-01-ablation-experiment-design.md`、`docs/superpowers/specs/2026-06-01-paper-rewrite-4.9-ablation.md` — 連動更新

---

## Task 1: precompute_class_presence.py（每影像類別表）

**Files:**
- Create: `segment-anything/scripts/precompute_class_presence.py`
- Test: `segment-anything/tests/test_precompute_class_presence.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_precompute_class_presence.py
"""
執行：python -m pytest segment-anything/tests/test_precompute_class_presence.py -v
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from scripts.precompute_class_presence import scan_gt_mask, build_class_presence


def test_scan_gt_mask_excludes_255(tmp_path):
    m = np.full((8, 8), 255, dtype=np.uint8)
    m[0:4, 0:4] = 3      # class 3
    m[4:8, 4:8] = 7      # class 7
    p = str(tmp_path / "gt.png")
    cv2.imwrite(p, m)
    present, counts = scan_gt_mask(p, num_classes=19)
    assert set(present) == {3, 7}                 # 255 excluded
    assert counts[3] == 16 and counts[7] == 16
    assert counts[0] == 0


def test_build_class_presence_over_csv(tmp_path):
    # 兩張合成 GT：img A 含 {0,18}，img B 含 {0}
    a = np.zeros((4, 4), dtype=np.uint8); a[0, 0] = 18
    b = np.zeros((4, 4), dtype=np.uint8)
    pa, pb = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    cv2.imwrite(pa, a); cv2.imwrite(pb, b)
    csv_p = str(tmp_path / "train.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gt_path"]); w.writerow([pa]); w.writerow([pb])
    out = str(tmp_path / "cp.json")
    data = build_class_presence(csv_p, out, num_classes=19)
    assert set(data["presence"][pa]) == {0, 18}
    assert set(data["presence"][pb]) == {0}
    # 全域像素計數：class 0 = 15(a) + 16(b) = 31；class 18 = 1
    assert data["class_pixel_counts"][18] == 1
    assert data["class_pixel_counts"][0] == 31
    assert os.path.isfile(out)
    with open(out) as f:
        assert json.load(f)["class_pixel_counts"][str(18)] == 1 or json.load(open(out))["class_pixel_counts"]["18"] == 1
```

- [ ] **Step 2: Run, verify FAIL**

Run: `python -m pytest segment-anything/tests/test_precompute_class_presence.py -v`
Expected: ImportError (module/functions absent).

- [ ] **Step 3: Implement**

```python
# segment-anything/scripts/precompute_class_presence.py
"""
掃描訓練集 GT 遮罩，建立「每影像含哪些類別」表與全域類別像素計數，供 RareClassSampler 使用。

用法：
  python scripts/precompute_class_presence.py \
    --csv /home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_train.csv \
    --out /home/rvl1421/SAM_research-1/Datasets/class_presence.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def scan_gt_mask(gt_path, num_classes=19):
    """回傳 (present_classes:list[int], counts:list[int] len=num_classes)。255/越界忽略。"""
    m = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"無法讀取 GT: {gt_path}")
    counts = [0] * num_classes
    vals, cnts = np.unique(m, return_counts=True)
    for v, c in zip(vals.tolist(), cnts.tolist()):
        if 0 <= v < num_classes:
            counts[v] = int(c)
    present = [c for c in range(num_classes) if counts[c] > 0]
    return present, counts


def build_class_presence(csv_path, out_path, num_classes=19):
    df = pd.read_csv(csv_path)
    if 'gt_path' not in df.columns:
        raise ValueError("CSV 缺少 gt_path 欄位")
    presence = {}
    total_counts = [0] * num_classes
    for gt in tqdm(df['gt_path'].tolist(), desc='scan GT'):
        present, counts = scan_gt_mask(gt, num_classes)
        presence[gt] = present
        for c in range(num_classes):
            total_counts[c] += counts[c]
    data = {
        'num_classes': num_classes,
        'presence': presence,
        'class_pixel_counts': {c: total_counts[c] for c in range(num_classes)},
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_classes', type=int, default=19)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if os.path.isfile(args.out) and not args.force \
            and os.path.getmtime(args.out) >= os.path.getmtime(args.csv):
        print(f"✅ 快取已是最新，略過：{args.out}（--force 可強制重建）")
        return
    data = build_class_presence(args.csv, args.out, args.num_classes)
    n = len(data['presence'])
    print(f"✅ 掃描 {n} 張 GT → {args.out}")
    print(f"   class_pixel_counts: {data['class_pixel_counts']}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run, verify PASS**

Run: `python -m pytest segment-anything/tests/test_precompute_class_presence.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/scripts/precompute_class_presence.py segment-anything/tests/test_precompute_class_presence.py
git commit -m "feat(rcs): add precompute_class_presence (per-image class table + pixel counts)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: RareClassSampler（忠實 DAFormer 取樣器）

**Files:**
- Create: `segment-anything/utils/rare_class_sampler.py`
- Test: `segment-anything/tests/test_rare_class_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_rare_class_sampler.py
"""
執行：python -m pytest segment-anything/tests/test_rare_class_sampler.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils.rare_class_sampler import RareClassSampler


def _toy():
    # 3 類；class 0 常見(在 idx 0..7)，class 1 中等(idx 8,9)，class 2 稀有(idx 9)
    presence = [[0]]*8 + [[0, 1], [1, 2]]   # 10 images, indices 0..9
    pixel_counts = [800, 20, 2]             # f: 0 大、2 極小
    return presence, pixel_counts


def test_class_probs_softmax_formula():
    presence, counts = _toy()
    s = RareClassSampler(presence, counts, num_samples=10, temperature=0.01,
                         seed=42, num_classes=3)
    f = torch.tensor(counts, dtype=torch.float64); f = f / f.sum()
    logits = (1.0 - f) / 0.01
    expected = torch.softmax(logits, dim=0)
    got = torch.tensor(s.class_probs, dtype=torch.float64)
    assert torch.allclose(got, expected, atol=1e-6)


def test_rare_class_drawn_far_more_than_uniform():
    presence, counts = _toy()
    s = RareClassSampler(presence, counts, num_samples=10, temperature=0.01,
                         seed=42, num_classes=3)
    draws = [s._draw_one_class() for _ in range(30000)]
    from collections import Counter
    cc = Counter(draws)
    # 稀有 class 2 的抽取率應遠高於 1/3（RCS 過取樣），且 > class 0
    assert cc[2] / 30000 > 0.30
    assert cc[2] > cc[0]


def test_iter_yields_valid_indices_and_length():
    presence, counts = _toy()
    s = RareClassSampler(presence, counts, num_samples=10, temperature=0.01,
                         seed=42, num_classes=3)
    idxs = list(iter(s))
    assert len(idxs) == 10
    assert all(0 <= i < 10 for i in idxs)
    # class 2 只在 idx 9 → 抽到 class 2 時必得 idx 9（含 idx 9 應頻繁出現）
    assert 9 in idxs


def test_reproducible_with_seed():
    presence, counts = _toy()
    a = list(iter(RareClassSampler(presence, counts, 200, 0.01, seed=123, num_classes=3)))
    b = list(iter(RareClassSampler(presence, counts, 200, 0.01, seed=123, num_classes=3)))
    c = list(iter(RareClassSampler(presence, counts, 200, 0.01, seed=999, num_classes=3)))
    assert a == b          # 同 seed 同序列
    assert a != c          # 不同 seed 不同序列


def test_empty_class_excluded():
    # class 1 無任何影像 → P(1)=0，不會被抽
    presence = [[0], [0], [2], [2]]
    counts = [100, 0, 5]      # class 1 計數 0 且無影像
    s = RareClassSampler(presence, counts, num_samples=100, temperature=0.01,
                         seed=7, num_classes=3)
    assert s.class_probs[1] == 0.0
    draws = [s._draw_one_class() for _ in range(2000)]
    assert 1 not in draws
```

- [ ] **Step 2: Run, verify FAIL**

Run: `python -m pytest segment-anything/tests/test_rare_class_sampler.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# segment-anything/utils/rare_class_sampler.py
"""
RareClassSampler — DAFormer 風格的稀有類別取樣（Hoyer et al., CVPR 2022）。

每抽一個樣本：
  1. 依 P(c) = softmax((1 - f(c)) / T) 抽一個類別 c（f(c) = 類別像素頻率）。
  2. 從「GT 含類別 c」的影像中均勻抽一張，回傳 dataset index。
無任何影像含之的類別，其 P(c) 設 0 並重正規化。所有隨機性由綁定 seed 的
torch.Generator 驅動，跨 run 可重現、跨 epoch 變化。
"""
from typing import List

import torch
from torch.utils.data import Sampler


class RareClassSampler(Sampler):
    def __init__(self, class_presence: List[List[int]], class_pixel_counts: List[int],
                 num_samples: int, temperature: float = 0.01, seed: int = 42,
                 num_classes: int = 19):
        """
        Args:
            class_presence: 長度 = dataset 大小；class_presence[i] = 第 i 張影像含的類別 id 清單。
            class_pixel_counts: 長度 = num_classes；各類別總像素數（precompute 提供）。
            num_samples: 每個 epoch 產生的 index 數（通常 = len(dataset)）。
            temperature, seed, num_classes: 見模組說明。
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.temperature = temperature
        self.seed = seed

        # f(c) 與 P(c)
        counts = torch.tensor(class_pixel_counts, dtype=torch.float64)
        f = counts / counts.sum().clamp(min=1.0)
        logits = (1.0 - f) / temperature
        probs = torch.softmax(logits, dim=0)

        # class → 含該類影像的 index 清單
        self.class_to_indices = {c: [] for c in range(num_classes)}
        for i, present in enumerate(class_presence):
            for c in present:
                if 0 <= c < num_classes:
                    self.class_to_indices[c].append(i)

        # 無影像的類別 P(c)=0，重正規化
        for c in range(num_classes):
            if len(self.class_to_indices[c]) == 0:
                probs[c] = 0.0
        probs = probs / probs.sum().clamp(min=1e-12)
        self.class_probs = probs.tolist()
        self._probs_t = probs

        # 持久 generator：__init__ 綁 seed，跨 __iter__ 沿用 → 可重現且逐 epoch 變化
        self._g = torch.Generator()
        self._g.manual_seed(seed)

    def _draw_one_class(self) -> int:
        return int(torch.multinomial(self._probs_t, 1, generator=self._g).item())

    def _draw_one_index(self, c: int) -> int:
        idxs = self.class_to_indices[c]
        j = int(torch.randint(len(idxs), (1,), generator=self._g).item())
        return idxs[j]

    def __iter__(self):
        for _ in range(self.num_samples):
            c = self._draw_one_class()
            yield self._draw_one_index(c)

    def __len__(self):
        return self.num_samples
```

- [ ] **Step 4: Run, verify PASS**

Run: `python -m pytest segment-anything/tests/test_rare_class_sampler.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/utils/rare_class_sampler.py segment-anything/tests/test_rare_class_sampler.py
git commit -m "feat(rcs): add RareClassSampler (faithful DAFormer class->image sampling)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: train.py 接 RCS（flags + sampler + config.json）

**Files:**
- Modify: `segment-anything/train.py`（imports、argparse、build sampler、DataLoader、config.json）

- [ ] **Step 1: 加 argparse flags**

在消融開關區（`--ref` 之後）加：
```python
    parser.add_argument("--rcs", action=argparse.BooleanOptionalAction, default=True,
                        help="Rare Class Sampling：依稀有類過取樣訓練影像（--no-rcs 關閉=純 shuffle）")
    parser.add_argument("--rcs_temp", type=float, default=0.01,
                        help="RCS 溫度 T（DAFormer 預設 0.01）")
    parser.add_argument("--class_presence", type=str, default=None,
                        help="class_presence.json 路徑（預設取 train_csv 同目錄）")
```

- [ ] **Step 2: imports + 建 sampler 並接上 train_loader**

頂部加：
```python
import json as _json  # 若檔案已 import json，沿用既有；否則新增 import json
from utils.rare_class_sampler import RareClassSampler
```
（注意：Task 7 已為 ablation 加過 `import json`；若已存在勿重複。）

把 train_loader 區塊（約 train.py:317-331，`loader_generator` 與 `DataLoader(... shuffle=True ...)`）改為：
```python
    # ★ DataLoader generator 綁定 seed，確保 shuffle / 取樣順序可重現
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_sampler = None
    if args.rcs:
        import json
        cp_path = args.class_presence or os.path.join(
            os.path.dirname(args.train_csv), "class_presence.json")
        if not os.path.isfile(cp_path):
            raise FileNotFoundError(
                f"--rcs 啟用但找不到 {cp_path}；請先執行 "
                f"scripts/precompute_class_presence.py（見 ABLATION_RUNBOOK Phase 0）。")
        with open(cp_path) as f:
            cp = json.load(f)
        num_classes = cp.get('num_classes', 19)
        # 依 train_ds 的 row 順序對齊 class_presence
        gt_paths = train_ds.data['gt_path'].tolist()
        presence_list = []
        for gp in gt_paths:
            if gp not in cp['presence']:
                raise KeyError(f"class_presence.json 缺少 {gp}；請以 --force 重新 precompute。")
            presence_list.append(cp['presence'][gp])
        pixel_counts = [cp['class_pixel_counts'][str(c)] if str(c) in cp['class_pixel_counts']
                        else cp['class_pixel_counts'].get(c, 0) for c in range(num_classes)]
        train_sampler = RareClassSampler(
            presence_list, pixel_counts, num_samples=len(train_ds),
            temperature=args.rcs_temp, seed=args.seed, num_classes=num_classes)
        print(f"[RCS] enabled (T={args.rcs_temp}); class_probs top5 rare = "
              f"{sorted(range(num_classes), key=lambda c: -train_sampler.class_probs[c])[:5]}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        collate_fn=WeatherSegmentationDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
```

- [ ] **Step 3: config.json 增記 rcs**

在寫 `ablation_config.json` 的 `json.dump({**abl_cfg, ...})` 內加入 `"rcs": args.rcs, "rcs_temp": args.rcs_temp`：
```python
    with open(os.path.join(args.output_dir, "ablation_config.json"), "w") as f:
        json.dump({**abl_cfg, "seed": args.seed,
                   "lovasz_weight": args.lovasz_weight,
                   "dice_weight": args.dice_weight,
                   "rcs": args.rcs, "rcs_temp": args.rcs_temp}, f, indent=2)
```

- [ ] **Step 4: 驗證**

- `python -m py_compile segment-anything/train.py && echo ok`
- `python segment-anything/train.py --help 2>&1 | grep -E "rcs|rcs_temp|class_presence"` → 3 flags 出現。
- 既有測試不受影響：`python -m pytest segment-anything/tests/ -v`（全綠）。
- （完整 RCS smoke 留 Task 6。）

- [ ] **Step 5: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/train.py
git commit -m "feat(rcs): wire --rcs/--rcs_temp into train.py (sampler + config dump)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: aggregate_ablation.py 支援 R8 / FULL=R8 / C2 獨立

**Files:**
- Modify: `segment-anything/scripts/aggregate_ablation.py`
- Modify: `segment-anything/tests/test_aggregate_ablation.py`

- [ ] **Step 1: 更新測試（先讀現有測試，調整累積表期望）**

READ `tests/test_aggregate_ablation.py`。把 `test_build_summary_table_has_rows_and_delta` 的 run 集改為 R1–R8（8 列），FULL=R8：
```python
def test_build_summary_table_r1_to_r8(tmp_path):
    root = str(tmp_path)
    seq = [('R1',0.40),('R2',0.55),('R3',0.58),('R4',0.60),
           ('R5',0.61),('R6',0.63),('R7',0.655),('R8',0.66)]
    for rid, m in seq:
        _write_run(root, rid, 42, m)   # _write_run 既有 helper
    runs = load_runs(root, results_filename='e1_results.json')
    tex = build_summary_table(runs)
    assert 'R1' in tex and 'R8' in tex
    # Δ vs FULL(=R8=0.66)：R1 ≈ (0.40-0.66)*100 = -26.0
    assert '-26.0' in tex
```
新增一個 loss 表測試，確認 C2 取自獨立 `C2` run（非 R6）：
```python
def test_loss_table_c2_uses_own_run(tmp_path):
    root = str(tmp_path)
    for rid, m in [('R8',0.66),('C1',0.62),('C2',0.64)]:
        _write_run(root, rid, 42, m)
    runs = load_runs(root, results_filename='e1_results.json')
    tex = build_loss_table(runs)
    assert 'C2' in tex
```

- [ ] **Step 2: Run, verify FAIL**

Run: `python -m pytest segment-anything/tests/test_aggregate_ablation.py -v`
Expected: 新測試 FAIL（summary 仍停在 R6/FULL；loss 表 C2 仍對應 R6）。

- [ ] **Step 3: 修改 aggregate_ablation.py**

READ 現有 `build_summary_table` / `build_loss_table`。改動：
- `build_summary_table`：cumulative 列序改為 `['R1','R2','R3','R4','R5','R6','R7','R8']`；FULL 參照（Δ 基準）改為 `runs['R8']`；R8 的 Δ 印 `---`（沿用既有 FULL-row 判斷，改判 `rid=='R8'`）。
- `build_adapter_table` / `build_loss_table`：FULL 參照由 `R8` 提供（取代舊 `FULL`/`R7`）；loss 表 C2 列改 `('C2', 'C2', False)`（用自身 `C2` run，**不再** 複用 `R6`）。
- 若程式內以字串 `'FULL'` 當 key，統一改為 `'R8'`（或讓 `_write_run`/run dir 命名以 `R8` 為 FULL）。**確認 run 目錄命名**：FULL 的 run dir 為 `R8_seed*`（見 Task 5 run_ablation.sh）。

- [ ] **Step 4: Run, verify PASS**

Run: `python -m pytest segment-anything/tests/test_aggregate_ablation.py -v`
Expected: 全綠（含既有 mean_std/fmt_cell 測試）。

- [ ] **Step 5: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/scripts/aggregate_ablation.py segment-anything/tests/test_aggregate_ablation.py
git commit -m "feat(rcs): aggregate summary R1-R8, FULL=R8, loss-table C2 standalone

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 連動文件更新（spec / run_ablation.sh / runbook / paper-rewrite）

**Files:**
- Modify: `segment-anything/run_ablation.sh`
- Modify: `segment-anything/ABLATION_RUNBOOK.md`
- Modify: `docs/superpowers/specs/2026-06-01-ablation-experiment-design.md`
- Modify: `docs/superpowers/specs/2026-06-01-paper-rewrite-4.9-ablation.md`

- [ ] **Step 1: run_ablation.sh**

READ 現檔。改動：
- 在所有 `python train.py` 前，先跑一次 precompute（冪等）：
```bash
python scripts/precompute_class_presence.py \
  --csv /home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_train.csv \
  --out /home/rvl1421/SAM_research-1/Datasets/class_presence.json
```
- seeds 改為**僅 FULL ×3**（`SEEDS_KEY` 只用於 FULL=R8；R1 改單 seed）。
- 每個 run 加 RCS flag：R1–R7、A1/A2/C1/C2 一律 `--no-rcs`；**新增 R7**（= 舊 FULL，`--no-rcs`，單 seed）與 **R8=FULL**（`--rcs`，3 seeds，output_dir `R8_seed*`）。
- **C2 改為獨立 run**：`--inject pre --decoder unified --lrh --no-mfb --lovasz_weight 1 --dice_weight 1 --no-rcs`（mfb off、rcs off），output_dir `C2_seed42`。
- A1/A2/C1 維持單一維度差異，且 `--no-rcs`（因它們相對「FULL 去 RCS=R7」基準做單維度比較；RCS 與這些維度正交，於正文說明）。
- eval 迴圈不變（掃 `outputs_ablation/*/`）；aggregate 不變。
- `bash -n run_ablation.sh` 通過。

> 說明列出 12 unique configs / 14 runs：R1(1)+R2(1)+R3(1)+R4(1)+R5(1)+R6(1)+R7(1)+R8(3)+A1(1)+A2(1)+C1(1)+C2(1)=14。

- [ ] **Step 2: ABLATION_RUNBOOK.md**

- Phase 0 新增勾項：「precompute class_presence」指令（同上）。
- Phase 2 FULL 指令：output_dir 改 `R8_seed42`、加 `--rcs`（或依預設 on）；說明 FULL=R8 含 RCS。
- run 矩陣/順序：更新為 12 configs / 14 runs、僅 FULL ×3、新增 R7/R8/RCS 說明。

- [ ] **Step 3: ablation-experiment-design.md（spec 修訂）**

- §0 表格範圍 / §1 run 矩陣：改為 12 configs / 14 runs、新增 R7（舊FULL,no-rcs）/R8(=FULL,+RCS)、seeds 僅 FULL ×3、C2 不再複用 R6（改獨立 run）、新增 `--rcs/--rcs_temp` 開關（標 train-only、eval 不受影響）。
- §8.5 模組化：補一句 RCS 取樣器為新增獨立模組（`utils/rare_class_sampler.py`），不影響既有模型/eval。

- [ ] **Step 4: paper-rewrite-4.9-ablation.md**

- §4 累積表：新增 R8（+RCS）列；RCS 貢獻 = R7→R8。
- 新增 RCS 方法敘述要點：P(c)=softmax((1−f(c))/T)、T=0.01、先抽類別再抽影像、f(c) 來源（1600-train precompute）、可重現性（seed）。
- 長尾證據：以 R7→R8 的 bus/moto/bicycle IoU 變化佐證 RCS（取代/補充原 MFB 段）。

- [ ] **Step 5: 驗證 + Commit**

- `bash -n segment-anything/run_ablation.sh && echo ok`
- `grep -c "weather_sam_best_latest.pth" segment-anything/run_ablation.sh` → 1
```bash
cd /home/rvl1421/SAM_research-1
git add -f segment-anything/run_ablation.sh segment-anything/ABLATION_RUNBOOK.md \
  docs/superpowers/specs/2026-06-01-ablation-experiment-design.md \
  docs/superpowers/specs/2026-06-01-paper-rewrite-4.9-ablation.md
git commit -m "docs(rcs): integrate RCS into ablation matrix, runbook, run script, paper-rewrite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 整合 smoke（RCS on 1 epoch）

**Files:** 無（驗證用，不改碼）

- [ ] **Step 1: precompute（真實 train CSV）**

Run:
```bash
cd /home/rvl1421/SAM_research-1/segment-anything
python scripts/precompute_class_presence.py \
  --csv /home/rvl1421/SAM_research-1/Datasets/acdc_adverse_ref_rgb_train.csv \
  --out /home/rvl1421/SAM_research-1/Datasets/class_presence.json
```
Expected: 印「掃描 1600 張 GT」+ class_pixel_counts（rider/moto/bike 計數極小）。

- [ ] **Step 2: RCS on 1-epoch smoke**

Run:
```bash
python train.py --epochs 1 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
  --inject pre --decoder unified --lrh --mfb --lovasz_weight 1 --dice_weight 1 \
  --rcs --seed 42 --output_dir /tmp/smoke_rcs
```
Expected: 印 `[RCS] enabled (T=0.01)...`、訓練跑通、`/tmp/smoke_rcs/ablation_config.json` 含 `"rcs": true`。

- [ ] **Step 3: 清理**

Run: `rm -rf /tmp/smoke_rcs`

> 無 commit（純驗證）。確認後即可依 runbook 跑正式 14 runs。

---

## 執行順序
1. Task 1–4（程式 + 測試，TDD）。
2. Task 5（文件連動）。
3. Task 6（precompute + RCS smoke）。
4. 依 runbook 跑 14 runs（FULL=R8 ×3，其餘 ×1）→ eval → aggregate → 驗證 R7→R8 的長尾提升 → 改寫論文。

## Self-Review 註記
- **Spec 覆蓋**：演算法(Task2)、precompute(Task1)、train 接線+config(Task3)、消融矩陣/R8/C2 獨立(Task4+5)、文件連動(Task5)、驗證(Task2/6) 全涵蓋。
- **型別一致**：`RareClassSampler(class_presence, class_pixel_counts, num_samples, temperature, seed, num_classes)`、`class_probs`、`_draw_one_class/_draw_one_index` 跨 Task2/3/測試一致；config.json key `rcs`/`rcs_temp` 跨 Task3 與 spec 一致；run dir `R8_seed*` 為 FULL 跨 Task4/5 一致。
- **執行期延展點**：Task4 需先讀現有 aggregator 列序/FULL-key 實作再改（已標 READ）；Task5 文件編輯需讀現檔再改（已標 READ）。
