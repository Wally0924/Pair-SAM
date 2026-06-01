# 消融實驗框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 為 WeatherSAM 加上 5 個消融開關（inject/decoder/lrh/mfb/ref）+ config 一致性機制 + 評估彙整，支撐論文 4.9 節 3 張表（累積 / adapter / loss），共 10 個 unique config、16 次訓練。

**Architecture:** 以既有 argparse 風格新增 flag；抽出 `assemble_semantic_logits` 共用函式消除 7 處 LRH 重複並讓 `--lrh` 單點開關；以 `build_weather_sam_from_config` 統一 train/eval 模型建構，配 `ablation_config.json` 杜絕 train/eval 不一致。所有開關預設值 = 現行 FULL 行為（向後相容）。

**Tech Stack:** PyTorch、SAM ViT-H、pytest（`conda run -n sam_env`）。

> 設計依據：[`docs/superpowers/specs/2026-06-01-ablation-experiment-design.md`](../specs/2026-06-01-ablation-experiment-design.md)。論文改寫見 [`docs/superpowers/specs/2026-06-01-paper-rewrite-4.9-ablation.md`](../specs/2026-06-01-paper-rewrite-4.9-ablation.md)。

---

## 檔案結構（決策鎖定）

**新增：**
- `segment_anything/modeling/semantic_assembly.py` — `assemble_semantic_logits()` 共用函式（19 類 scatter + gated LRH 單一來源）。
- `segment-anything/scripts/aggregate_ablation.py` — 掃 10 config 的 metrics JSON，吐 3 張表 `.tex`。
- `segment-anything/run_ablation.sh` — 16 條訓練 + eval + 彙整指令，釘死每 run 的 flag/seed。
- `segment-anything/tests/test_semantic_assembly.py`、`test_decoder_per_class.py`、`test_ref_switch.py`、`test_mfb_switch.py`、`test_build_from_config.py`、`test_aggregate_ablation.py`。

**修改：**
- `segment_anything/modeling/weather_mask_decoder.py` — `decoder_mode` 屬性 + `predict_masks_per_class()`。
- `segment_anything/modeling/weather_sam.py` — `use_lrh` 屬性（建構/forward 路徑不直接套 LRH，維持外部套用）。
- `segment_anything/modeling/vgg_adapter.py` — `use_reference` 屬性 + `_inject_at_stage` 零張量路徑。
- `utils/new_loss.py` — `ContextLoss(use_mfb=...)`。
- `segment_anything/build_weather_sam.py` — `build_weather_sam_from_config()`。
- `weather_trainer.py` — train/validate 兩處改用 helper；MFB 開關套用至 `_mask_cls_w`。
- `train.py` — 5 個 flag + 寫 `ablation_config.json`。
- `scripts/eval/eval_e1_acdc_val_full.py`、`scripts/eval/_eval_common.py` — config-aware 建模 + 改用 helper。

**不動：** 1061 行 `weather_trainer.py` 的整體結構不重構；非消融路徑的 eval/viz/test_inference 留待 Task 11（可選）。

---

## Task 1: `assemble_semantic_logits` 共用函式（模組化 A）

**Files:**
- Create: `segment-anything/segment_anything/modeling/semantic_assembly.py`
- Test: `segment-anything/tests/test_semantic_assembly.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_semantic_assembly.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_semantic_assembly.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits


def test_scatter_places_classes_and_fills_rest():
    low_res = torch.randn(2, 4, 4)          # K=2, 4x4
    class_ids = [3, 7]
    out = assemble_semantic_logits(low_res, class_ids, fusion_head=None,
                                   num_classes=19, use_lrh=False, fill_value=-10.0)
    assert out.shape == (1, 19, 4, 4)
    assert torch.allclose(out[0, 3], low_res[0])
    assert torch.allclose(out[0, 7], low_res[1])
    # 未出現的類別維持填充值
    assert torch.allclose(out[0, 0], torch.full((4, 4), -10.0))


def test_use_lrh_false_skips_fusion_head():
    low_res = torch.randn(1, 4, 4)
    head = nn.Identity()  # 若被呼叫會是 no-op，但我們驗證「沒被呼叫」的等價輸出
    out_off = assemble_semantic_logits(low_res, [0], fusion_head=head,
                                       num_classes=19, use_lrh=False)
    out_raw = assemble_semantic_logits(low_res, [0], fusion_head=None,
                                       num_classes=19, use_lrh=False)
    assert torch.allclose(out_off, out_raw)


def test_use_lrh_true_applies_fusion_head():
    low_res = torch.randn(1, 4, 4)

    class AddOne(nn.Module):
        def forward(self, x):
            return x + 1.0

    out_on = assemble_semantic_logits(low_res, [0], fusion_head=AddOne(),
                                      num_classes=19, use_lrh=True)
    out_raw = assemble_semantic_logits(low_res, [0], fusion_head=None,
                                       num_classes=19, use_lrh=False)
    assert torch.allclose(out_on, out_raw + 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_semantic_assembly.py -v`
Expected: FAIL — `ModuleNotFoundError: ... semantic_assembly`

- [ ] **Step 3: Write minimal implementation**

```python
# segment-anything/segment_anything/modeling/semantic_assembly.py
"""
語意 logits 組裝共用函式。

歷史背景：原本「將 K 個 active class 的 low-res logits scatter 進 19 類別張量
（缺席類別填 -10.0），再選擇性套用 context_fusion_head (LRH)」這段邏輯散落在
trainer(train/validate)、eval、viz、inference 共 7 處。為了讓 --lrh 開關成為單一
真值來源、並杜絕 train/eval 套用不一致，集中於此。
"""
from typing import List, Optional

import torch
import torch.nn as nn


def assemble_semantic_logits(
    low_res_logits: torch.Tensor,      # (K, H, W) — 每個 active class 一張 logit map
    class_ids: List[int],              # len=K，對應 0..num_classes-1
    fusion_head: Optional[nn.Module],  # ResidualDWConvFusion；use_lrh=False 時可為 None
    *,
    num_classes: int = 19,
    use_lrh: bool = True,
    fill_value: float = -10.0,
) -> torch.Tensor:
    """組裝 (1, num_classes, H, W) 語意 logits，並依 use_lrh 決定是否套用 LRH。

    Returns:
        (1, num_classes, H, W) logits（use_lrh=True 時為 LRH 精修後）。
    """
    K, H, W = low_res_logits.shape
    full = torch.full(
        (1, num_classes, H, W), fill_value,
        device=low_res_logits.device, dtype=low_res_logits.dtype,
    )
    for k, c in enumerate(class_ids):
        full[0, c] = low_res_logits[k]

    if use_lrh:
        if fusion_head is None:
            raise ValueError("use_lrh=True 但 fusion_head 為 None")
        return fusion_head(full)
    return full
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_semantic_assembly.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/semantic_assembly.py segment-anything/tests/test_semantic_assembly.py
git commit -m "feat(ablation): add assemble_semantic_logits shared helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `--lrh` 開關（model.use_lrh + 遷移 3 個消融路徑呼叫點）

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_sam.py`（加 `self.use_lrh = True`）
- Modify: `segment-anything/weather_trainer.py:426-429`（train）、`:932-935`（validate）
- Modify: `segment-anything/scripts/eval/eval_e1_acdc_val_full.py:62-68`

- [ ] **Step 1: 加 `use_lrh` 屬性**

在 `WeatherSAM.__init__`（[weather_sam.py](../../segment-anything/segment_anything/modeling/weather_sam.py) 約 line 58-60，`self.num_classes` 附近）加：

```python
        self.num_classes = num_classes
        # [ablation] LRH (context_fusion_head) 是否套用；外部組裝點依此 gate。預設 True = FULL 行為。
        self.use_lrh = True
```

- [ ] **Step 2: 遷移 trainer train 路徑**

[weather_trainer.py](../../segment-anything/weather_trainer.py) 頂部 import：

```python
from segment_anything.modeling.semantic_assembly import assemble_semantic_logits
```

把 train 段（約 426-429）：

```python
                    full_class_logits = selected_logits.unsqueeze(0)  # (1, 19, 256, 256)

                    fused_logits = self.model.context_fusion_head(full_class_logits)
```

改為（保留 `full_class_logits` 供 head_delta_norm 監控使用）：

```python
                    full_class_logits = selected_logits.unsqueeze(0)  # (1, 19, 256, 256)

                    fused_logits = assemble_semantic_logits(
                        selected_logits, class_ids_out,
                        fusion_head=self.model.context_fusion_head,
                        num_classes=self.model.num_classes,
                        use_lrh=self.model.use_lrh,
                    )
```

> 註：`selected_logits` 為 (19, 256, 256) 還是 (K, 256, 256)？確認其 shape。若已是 19 類密集張量則 `class_ids_out` 為 0..18；若為 K 類稀疏則照 helper 語意。實作時 **先讀 411-429 上下文確認 `selected_logits` 與 `class_ids_out` 對應關係**，使 helper 輸出與原 `full_class_logits` 等價。

- [ ] **Step 3: 遷移 trainer validate 路徑**

同樣改 validate 段（約 932-935）為 `assemble_semantic_logits(...)`，參數同上。

- [ ] **Step 4: 遷移 eval_e1 路徑**

[eval_e1_acdc_val_full.py](../../segment-anything/scripts/eval/eval_e1_acdc_val_full.py) 頂部 import helper；把 62-68 的 `torch.full(...)` + 迴圈 + `model.context_fusion_head(full)` 改為：

```python
            fused = assemble_semantic_logits(
                low_res, class_ids_out,
                fusion_head=model.context_fusion_head,
                num_classes=NUM_CLASSES,
                use_lrh=model.use_lrh,
            )
```

- [ ] **Step 5: 驗證等價（無迴歸）**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/ -v`
接著手動 smoke（確認 import 與 forward 不爆）：
Run: `conda run -n sam_env python -c "import sys; sys.path.insert(0,'segment-anything'); from segment_anything.modeling.semantic_assembly import assemble_semantic_logits; print('ok')"`
Expected: `ok`，且既有測試全 PASS。

- [ ] **Step 6: Commit**

```bash
git add segment-anything/segment_anything/modeling/weather_sam.py segment-anything/weather_trainer.py segment-anything/scripts/eval/eval_e1_acdc_val_full.py
git commit -m "refactor(ablation): route LRH through assemble_semantic_logits, add model.use_lrh

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `--decoder per_class` 開關（MaskDecoder 逐類別解碼）

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_mask_decoder.py`
- Test: `segment-anything/tests/test_decoder_per_class.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_decoder_per_class.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_decoder_per_class.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from segment_anything.modeling.weather_mask_decoder import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer


def _make_decoder(num_classes=4):
    tf = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=512)
    dec = MaskDecoder(transformer_dim=256, transformer=tf, num_classes=num_classes)
    dec.eval()
    return dec


def _inputs(K=2):
    img = torch.randn(1, 256, 64, 64)
    pe = torch.randn(1, 256, 64, 64)
    sparse = torch.randn(K, 2, 256)      # K classes, N_tok=2
    dense = torch.randn(1, 256, 64, 64)
    class_ids = list(range(K))
    return img, pe, sparse, dense, class_ids


def test_default_mode_is_unified():
    dec = _make_decoder()
    assert dec.decoder_mode == 'unified'


def test_param_count_identical_across_modes():
    dec = _make_decoder()
    n = sum(p.numel() for p in dec.parameters())
    dec.decoder_mode = 'per_class'
    assert sum(p.numel() for p in dec.parameters()) == n  # mode 僅切換流程，不增減參數


def test_per_class_isolates_classes_unified_does_not():
    dec = _make_decoder()
    img, pe, sparse, dense, class_ids = _inputs(K=2)
    sparse2 = sparse.clone()
    sparse2[1] += 5.0  # 只擾動 class 1 的 prompt

    with torch.no_grad():
        dec.decoder_mode = 'per_class'
        a = dec.forward_semantic(img, pe, sparse, dense, class_ids)
        b = dec.forward_semantic(img, pe, sparse2, dense, class_ids)
        # per-class：class 0 不受 class 1 擾動影響
        assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)

        dec.decoder_mode = 'unified'
        c = dec.forward_semantic(img, pe, sparse, dense, class_ids)
        d = dec.forward_semantic(img, pe, sparse2, dense, class_ids)
        # unified：跨類別 self-attention 使 class 0 受影響
        assert not torch.allclose(c[:, 0], d[:, 0], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_decoder_per_class.py -v`
Expected: FAIL — `AttributeError: 'MaskDecoder' object has no attribute 'decoder_mode'`

- [ ] **Step 3: Write minimal implementation**

[weather_mask_decoder.py](../../segment-anything/segment_anything/modeling/weather_mask_decoder.py)：`__init__` 末尾（line 63 後）加：

```python
        # [ablation] 'unified' = 所有 class query 同序列（跨類別 self-attention）；
        # 'per_class' = 每類別獨立 forward（移除跨類別交互）。預設 unified = FULL 行為。
        self.decoder_mode = 'unified'
```

`forward_semantic`（line 90-94）改為依 mode 分派：

```python
        if self.decoder_mode == 'per_class':
            return self.predict_masks_per_class(
                image_embeddings, image_pe,
                sparse_prompt_embeddings, dense_prompt_embeddings,
                class_ids,
            )
        return self.predict_masks_semantic(
            image_embeddings, image_pe,
            sparse_prompt_embeddings, dense_prompt_embeddings,
            class_ids,
        )
```

在 `predict_masks_semantic` 之後新增（複用 predict_masks_semantic 以 K=1 逐類別呼叫，保證參數與權重完全相同、且類別間無交互）：

```python
    def predict_masks_per_class(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        class_ids: List[int],
    ) -> torch.Tensor:
        """[ablation] 逐類別獨立解碼：每個 class 以 K=1 單獨跑一次 transformer，
        移除跨類別 self-attention。複用 predict_masks_semantic，參數量與統一查詢版相同。
        """
        masks = []
        for i, cls_id in enumerate(class_ids):
            mask_i = self.predict_masks_semantic(
                image_embeddings, image_pe,
                sparse_prompt_embeddings[i:i + 1],   # (1, N_tok, 256)
                dense_prompt_embeddings,
                [cls_id],
            )  # (1, 1, 256, 256)
            masks.append(mask_i)
        return torch.cat(masks, dim=1)  # (1, K, 256, 256)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_decoder_per_class.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/weather_mask_decoder.py segment-anything/tests/test_decoder_per_class.py
git commit -m "feat(ablation): add per-class decoder mode to MaskDecoder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `--ref off` 開關（injector 零張量移除 reference）

**Files:**
- Modify: `segment-anything/segment_anything/modeling/vgg_adapter.py`
- Test: `segment-anything/tests/test_ref_switch.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_ref_switch.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_ref_switch.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector


def _make_injector():
    inj = MultiScaleCrossAttnInjector()
    inj.eval()
    return inj


def _feats(B=1, H=16, W=16):
    # l2: stride8 256ch, l3: stride16 512ch（符合 fusion.pre_align 輸出維度）
    return {
        'l2': torch.randn(B, 256, H, W),
        'l3': torch.randn(B, 512, H, W),
    }


def test_use_reference_default_true():
    inj = _make_injector()
    assert inj.use_reference is True


def test_ref_off_insensitive_to_reference_content():
    inj = _make_injector()
    inj.use_reference = False
    out = torch.randn(1, 16, 16, inj.vit_dim if hasattr(inj, 'vit_dim') else 1280)
    # 餵兩組不同 reference 特徵，ref off 時注入結果應相同
    with torch.no_grad():
        inj.set_features(_feats()); a = inj._inject_at_stage(out.clone(), 0)
        inj.set_features(_feats()); b = inj._inject_at_stage(out.clone(), 0)
    assert torch.allclose(a, b, atol=1e-5)


def test_ref_on_sensitive_to_reference_content():
    inj = _make_injector()
    inj.use_reference = True
    out = torch.randn(1, 16, 16, inj.vit_dim if hasattr(inj, 'vit_dim') else 1280)
    with torch.no_grad():
        inj.set_features(_feats()); a = inj._inject_at_stage(out.clone(), 0)
        inj.set_features(_feats()); b = inj._inject_at_stage(out.clone(), 0)
    assert not torch.allclose(a, b, atol=1e-5)
```

> 實作前 **先讀 [vgg_adapter.py:44-101](../../segment-anything/segment_anything/modeling/vgg_adapter.py) 的 `__init__`** 確認 `MultiScaleCrossAttnInjector()` 預設可無參建構、`set_features` 的 key（`l2`/`l3`）、以及 ViT 通道維度屬性名，據此修正測試中 `out` 的 channel 與 feats 維度。

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_ref_switch.py -v`
Expected: FAIL — `AttributeError: ... 'use_reference'`

- [ ] **Step 3: Write minimal implementation**

`MultiScaleCrossAttnInjector.__init__` 末尾（約 line 94 附近）加：

```python
        # [ablation] 是否引入 reference 資訊。False 時將餵入 K/V 的 reference 特徵
        # 以零張量取代，保持 adapter 結構與參數量不變，僅移除參考內容。預設 True = FULL。
        self.use_reference = True
```

在 `_inject_at_stage` 內、`f_flat` 投影成 K/V 之前（[vgg_adapter.py:176](../../segment-anything/segment_anything/modeling/vgg_adapter.py) `f_flat = ...` 之後、line 179 `K = self.k_projs...` 之前）加：

```python
        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, P * P, -1)    # (B, P², kv_in)

        # [ablation] --ref off：移除 reference 內容（零張量），保留 adapter 容量
        if not self.use_reference:
            f_flat = torch.zeros_like(f_flat)

        K = self.k_projs[stage_idx](f_flat)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_ref_switch.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/vgg_adapter.py segment-anything/tests/test_ref_switch.py
git commit -m "feat(ablation): add use_reference switch (zero-tensor) to injector

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `--mfb off` 開關（ContextLoss + trainer 類別權重 uniform）

**Files:**
- Modify: `segment-anything/utils/new_loss.py`（`ContextLoss`）
- Modify: `segment-anything/weather_trainer.py:117`（`_mask_cls_w`）
- Test: `segment-anything/tests/test_mfb_switch.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_mfb_switch.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_mfb_switch.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.new_loss import ContextLoss


def test_mfb_on_uses_nonuniform_weights():
    loss = ContextLoss(use_mfb=True)
    w = loss.ce_loss_fn.weight
    assert w is not None
    assert not torch.allclose(w, torch.ones_like(w))  # MFB 權重非均勻


def test_mfb_off_uses_uniform_weights():
    loss = ContextLoss(use_mfb=False)
    w = loss.ce_loss_fn.weight
    # uniform：weight=None 或全 1
    assert w is None or torch.allclose(w, torch.ones_like(w))


def test_mfb_default_on_backward_compatible():
    loss = ContextLoss()  # 預設 = FULL 行為
    w = loss.ce_loss_fn.weight
    assert w is not None and not torch.allclose(w, torch.ones_like(w))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_mfb_switch.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'use_mfb'`

- [ ] **Step 3: Write minimal implementation**

[new_loss.py](../../segment-anything/utils/new_loss.py) `ContextLoss.__init__`（line 126-138）改為接受 `use_mfb`：

```python
    def __init__(self, ce_weight: float = 1.0, num_classes: int = 19,
                 label_smoothing: float = 0.0, lovasz_weight: float = 0.0,
                 use_mfb: bool = True):
        super().__init__()
        self.ce_weight     = ce_weight
        self.lovasz_weight = lovasz_weight
        self.use_mfb       = use_mfb

        class_weights = _build_median_freq_weights(_ACDC_CLASS_FREQ)
        self.register_buffer('class_weights', class_weights)
        ce_weight_arg = self.class_weights if use_mfb else None  # [ablation] off = uniform
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=ce_weight_arg, ignore_index=255,
            label_smoothing=label_smoothing,
        )
        self.ce_unweighted = nn.CrossEntropyLoss(ignore_index=255)
```

[weather_trainer.py](../../segment-anything/weather_trainer.py) 約 line 95-117：建構 `ContextLoss` 處傳入 `use_mfb=getattr(args, 'mfb', True)`；`_mask_cls_w` 改為：

```python
        use_mfb = getattr(args, 'mfb', True)
        self._mask_cls_w = (ACDC_CLASS_WEIGHTS if use_mfb
                            else torch.ones_like(ACDC_CLASS_WEIGHTS)).to(self.device)
```

> 實作前讀 95-117 確認 `ContextLoss(...)` 實際建構行，補上 `use_mfb=use_mfb`。

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_mfb_switch.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add segment-anything/utils/new_loss.py segment-anything/weather_trainer.py segment-anything/tests/test_mfb_switch.py
git commit -m "feat(ablation): add use_mfb switch (uniform weights) to loss

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `build_weather_sam_from_config` 統一建構（模組化 B）

**Files:**
- Modify: `segment-anything/segment_anything/build_weather_sam.py`
- Test: `segment-anything/tests/test_build_from_config.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_build_from_config.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_build_from_config.py -v
注意：此測試不載入 SAM checkpoint（checkpoint=None），僅驗證 config→屬性映射。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from segment_anything.build_weather_sam import build_weather_sam_from_config


def _cfg(**over):
    base = dict(model_type='vit_b', use_vgg_adapter=True, inject='pre',
                decoder='unified', lrh=True, mfb=True, ref=True)
    base.update(over)
    return base


def test_config_maps_to_attributes():
    m = build_weather_sam_from_config(_cfg(decoder='per_class', lrh=False, ref=False),
                                      checkpoint=None)
    assert m.mask_decoder.decoder_mode == 'per_class'
    assert m.use_lrh is False
    assert m.vgg_injector.use_reference is False


def test_full_defaults_backward_compatible():
    m = build_weather_sam_from_config(_cfg(), checkpoint=None)
    assert m.mask_decoder.decoder_mode == 'unified'
    assert m.use_lrh is True
    assert m.vgg_injector.use_reference is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_build_from_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_weather_sam_from_config'`

- [ ] **Step 3: Write minimal implementation**

[build_weather_sam.py](../../segment-anything/segment_anything/build_weather_sam.py) 新增（沿用既有 `build_weather_sam_vit_b/h`）：

```python
def build_weather_sam_from_config(cfg: dict, checkpoint=None):
    """[ablation] 依 config dict 建構 WeatherSAM，統一 train 與 eval 的建模路徑。

    cfg keys: model_type, use_vgg_adapter(bool), inject('pre'/'post'),
              decoder('unified'/'per_class'), lrh(bool), mfb(bool), ref(bool)
    """
    ckpt = checkpoint
    if cfg.get('model_type', 'vit_h') == 'vit_b':
        model = build_weather_sam_vit_b(checkpoint=ckpt)
    else:
        model = build_weather_sam_vit_h(checkpoint=ckpt)

    model.use_lrh = bool(cfg.get('lrh', True))
    model.mask_decoder.decoder_mode = cfg.get('decoder', 'unified')
    model.vgg_injector.use_reference = bool(cfg.get('ref', True))

    if cfg.get('use_vgg_adapter', True):
        model.enable_vgg_adapter(mode=cfg.get('inject', 'pre'))

    return model
```

> 確認 `WeatherSAM` 上 `mask_decoder`、`vgg_injector` 的屬性名（讀 [weather_sam.py:58-86](../../segment-anything/segment_anything/modeling/weather_sam.py)）。`--mfb` 屬於 loss 端（Task 5），不在模型建構，故不在此設定。

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_build_from_config.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/build_weather_sam.py segment-anything/tests/test_build_from_config.py
git commit -m "feat(ablation): add build_weather_sam_from_config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: train.py 接 5 個 flag + 寫 `ablation_config.json`

**Files:**
- Modify: `segment-anything/train.py`

- [ ] **Step 1: 加 argparse flag**

在 [train.py](../../segment-anything/train.py) `main()` 的 argparse 區（約 line 230-236，`--use_vgg_adapter` 附近）加：

```python
    # --- [ablation] 消融開關（預設值 = FULL，向後相容）---
    parser.add_argument("--inject", choices=["pre", "post"], default="pre",
                        help="WarpedVGG Adapter 注入位置：pre=block 自注意力前；post=後")
    parser.add_argument("--decoder", choices=["unified", "per_class"], default="unified",
                        help="解碼模式：unified=統一查詢；per_class=逐類別獨立解碼")
    parser.add_argument("--lrh", action=argparse.BooleanOptionalAction, default=True,
                        help="是否套用 LRH (ResidualDWConvFusion)")
    parser.add_argument("--mfb", action=argparse.BooleanOptionalAction, default=True,
                        help="CE/dice 是否套用 MFB 類別權重（--no-mfb = uniform）")
    parser.add_argument("--ref", action=argparse.BooleanOptionalAction, default=True,
                        help="Adapter 是否引入 reference K/V（--no-ref = 零張量）")
```

- [ ] **Step 2: 用 builder 建模 + 設開關**

把 line 250-260 的 `build_weather_sam_vit_h/b` + `enable_vgg_adapter()` 區塊改為以 config 建構：

```python
    abl_cfg = dict(
        model_type=args.model_type,
        use_vgg_adapter=args.use_vgg_adapter,
        inject=args.inject,
        decoder=args.decoder,
        lrh=args.lrh,
        mfb=args.mfb,
        ref=args.ref,
    )
    from segment_anything.build_weather_sam import build_weather_sam_from_config
    model = build_weather_sam_from_config(abl_cfg, checkpoint=model_checkpoint)
    print(f"[Ablation] config = {abl_cfg}")
```

> 移除原本獨立呼叫 `model.enable_vgg_adapter()` 的行（builder 已處理），避免重複註冊 hook。

- [ ] **Step 3: 寫 `ablation_config.json`**

`os.makedirs(args.output_dir, exist_ok=True)` 之後加：

```python
    import json
    _cfg_dump = dict(abl_cfg) if 'abl_cfg' in dir() else {}
```

實作時改為在 `abl_cfg` 定義後落地（含 seed 與 loss 權重，供 eval 重建與審計）：

```python
    with open(os.path.join(args.output_dir, "ablation_config.json"), "w") as f:
        json.dump({**abl_cfg, "seed": args.seed,
                   "lovasz_weight": args.lovasz_weight,
                   "dice_weight": args.dice_weight}, f, indent=2)
```

- [ ] **Step 4: 驗證 flag 解析與 config 落地（0-epoch smoke 之一）**

Run（vit_b 較快，不需真資料即可驗證 argparse 與 config 落地，遇缺資料報錯前 config 已寫出；若需可加 `--epochs 0`）：
`conda run -n sam_env python -c "import sys; sys.path.insert(0,'segment-anything'); sys.argv=['t','--help']; exec(open('segment-anything/train.py').read())" 2>&1 | grep -E "inject|decoder|lrh|mfb|ref"`
Expected: 5 個 flag 出現在 help。

- [ ] **Step 5: Commit**

```bash
git add segment-anything/train.py
git commit -m "feat(ablation): wire 5 ablation flags + dump ablation_config.json

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: eval_e1 config-aware（讀 config.json 重建模型）

**Files:**
- Modify: `segment-anything/scripts/eval/_eval_common.py`（`load_weather_sam_model`）
- Modify: `segment-anything/scripts/eval/eval_e1_acdc_val_full.py`（加 `--ckpt`/`--config`）

- [ ] **Step 1: `_eval_common` 加 config-aware 建模**

在 [_eval_common.py](../../segment-anything/scripts/eval/_eval_common.py) 新增（保留原 `load_weather_sam_model` 簽名以免破壞其他 eval）：

```python
import json

def load_weather_sam_from_ablation(ckpt_path, config_path=None, device='cuda'):
    """依 ablation_config.json 重建模型並載入 ckpt，確保與訓練 config 完全一致。"""
    from segment_anything.build_weather_sam import build_weather_sam_from_config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(ckpt_path), 'ablation_config.json')
    with open(config_path) as f:
        cfg = json.load(f)
    model = build_weather_sam_from_config(cfg, checkpoint=None)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state, strict=False)
    return model.to(device).eval(), cfg
```

> 確認 ckpt 儲存格式（`state['model']` 或直接 state_dict）——讀 trainer 儲存 checkpoint 的程式碼確認 key。

- [ ] **Step 2: eval_e1 接受 `--ckpt`/`--config`**

[eval_e1_acdc_val_full.py](../../segment-anything/scripts/eval/eval_e1_acdc_val_full.py) `main()` 加 argparse：

```python
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--out', default=None, help='輸出 JSON 路徑（預設依 ckpt 命名）')
    args = ap.parse_args()
    model, cfg = load_weather_sam_from_ablation(args.ckpt, args.config, device=DEVICE)
```

並把第 40 行原 `load_weather_sam_model(DEFAULT_CKPT, ...)` 改用上面的 `model`；輸出 JSON 路徑改用 `args.out`（或依 ckpt 目錄）以免覆寫。`model.use_lrh` 已由 cfg 設定，Task 2 的 helper 會正確 gate。

- [ ] **Step 3: 驗證**

Run（需先有任一 run 的 ckpt + ablation_config.json；若尚無，延後至 Task 10 smoke 一併驗證）：
`conda run -n sam_env python segment-anything/scripts/eval/eval_e1_acdc_val_full.py --ckpt <某run>/best.pth`
Expected: 產出該 run 的 per-class×per-condition JSON。

- [ ] **Step 4: Commit**

```bash
git add segment-anything/scripts/eval/_eval_common.py segment-anything/scripts/eval/eval_e1_acdc_val_full.py
git commit -m "feat(ablation): make eval_e1 config-aware via ablation_config.json

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: `aggregate_ablation.py`（彙整 3 張表 .tex）

**Files:**
- Create: `segment-anything/scripts/aggregate_ablation.py`
- Test: `segment-anything/tests/test_aggregate_ablation.py`

- [ ] **Step 1: Write the failing test**

```python
# segment-anything/tests/test_aggregate_ablation.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_aggregate_ablation.py -v
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.aggregate_ablation import mean_std, fmt_cell


def test_mean_std_single_value():
    m, s = mean_std([0.6568])
    assert abs(m - 0.6568) < 1e-9
    assert s == 0.0


def test_mean_std_three_seeds():
    m, s = mean_std([0.64, 0.65, 0.66])
    assert abs(m - 0.65) < 1e-9
    assert s > 0.0


def test_fmt_cell_percent_one_decimal():
    assert fmt_cell(0.6568) == "65.7"            # 單 seed → 一位小數
    assert fmt_cell(0.65, 0.01).startswith("65.0")  # 多 seed → mean±std
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_aggregate_ablation.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.aggregate_ablation`

- [ ] **Step 3: Write minimal implementation**

```python
# segment-anything/scripts/aggregate_ablation.py
"""
彙整消融實驗 metrics JSON → 3 張表的 LaTeX 片段。

用法：
  python scripts/aggregate_ablation.py --runs_root outputs_ablation --out tables.tex

每個 run 目錄需含 eval_e1 輸出的 JSON（overall_miou / per_condition_miou / per_class_iou）。
run 對應由 ablation_config.json 自動判別（依 decoder/lrh/mfb/ref/inject/use_vgg_adapter）。
"""
import argparse
import json
import math
import os
from typing import List, Tuple


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def fmt_cell(mean: float, std: float = 0.0) -> str:
    """mIoU 以百分比呈現；多 seed 時加 ±std。"""
    if std and std > 0.0:
        return f"{mean*100:.1f}$\\pm${std*100:.1f}"
    return f"{mean*100:.1f}"


def _load_runs(runs_root: str) -> dict:
    """掃 runs_root 下每個子目錄的 ablation_config.json + eval JSON。"""
    runs = {}
    for name in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, name)
        cfg_p = os.path.join(d, 'ablation_config.json')
        if not os.path.isfile(cfg_p):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        runs[name] = {'cfg': cfg, 'dir': d}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_root', required=True)
    ap.add_argument('--out', default='ablation_tables.tex')
    args = ap.parse_args()
    runs = _load_runs(args.runs_root)
    # 完整彙整邏輯（依 cfg 分組到 3 張表、跨 seed 平均）於執行時依實際 JSON 欄位完成。
    print(f"Loaded {len(runs)} runs from {args.runs_root}")
    # TODO(執行期)：依 spec §4 產出 tab:ablation_summary / tab:adapter_ablation / tab:loss_ablation
    with open(args.out, 'w') as f:
        f.write("% generated by aggregate_ablation.py\n")


if __name__ == '__main__':
    main()
```

> 註：表格組裝細節（cfg→表分組、欄位順序）依 eval JSON 實際 schema 於執行期補完；`mean_std`/`fmt_cell` 為已測試的核心工具，確保數值格式正確。**此為唯一允許的執行期延展點**，因表格欄位依賴實測 JSON。

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_aggregate_ablation.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add segment-anything/scripts/aggregate_ablation.py segment-anything/tests/test_aggregate_ablation.py
git commit -m "feat(ablation): add aggregate_ablation table builder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: `run_ablation.sh`（16 條訓練 + eval + 彙整）

**Files:**
- Create: `segment-anything/run_ablation.sh`

- [ ] **Step 1: 寫指令腳本**

```bash
# segment-anything/run_ablation.sh
#!/usr/bin/env bash
# 消融實驗 10 unique config / 16 訓練 run。R1/FULL/A2 各 3 seeds。
# C2 = R6（複用，不另訓）。每行一個 run，互相獨立可平行。
set -e
cd "$(dirname "$0")"
SEEDS_KEY="42 1234 2026"   # R1 / FULL / A2 用
COMMON="--epochs 80 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5"

run () { conda run -n sam_env python train.py $COMMON "$@"; }

# ── 累積表 R1–R6（單 seed=42，FULL 另跑 3 seeds 於下方）──
# R1 baseline：無 adapter / per-class / 純CE / 無LRH / 無MFB
for s in $SEEDS_KEY; do
  run --seed $s --no-use_vgg_adapter --decoder per_class --no-lrh --no-mfb \
      --lovasz_weight 0 --dice_weight 0 --output_dir outputs_ablation/R1_seed$s
done
# R2 +Ref 後置
run --seed 42 --inject post --decoder per_class --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir outputs_ablation/R2_seed42
# R3 前置注入
run --seed 42 --inject pre --decoder per_class --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir outputs_ablation/R3_seed42
# R4 統一查詢
run --seed 42 --inject pre --decoder unified --no-lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir outputs_ablation/R4_seed42
# R5 +LRH
run --seed 42 --inject pre --decoder unified --lrh --no-mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir outputs_ablation/R5_seed42
# R6 +Lovász/Dice（= loss 表的「取消 MFB」C2）
run --seed 42 --inject pre --decoder unified --lrh --no-mfb \
    --lovasz_weight 1 --dice_weight 1 --output_dir outputs_ablation/R6_seed42

# ── FULL（3 seeds）= R6 + MFB ──
for s in $SEEDS_KEY; do
  run --seed $s --inject pre --decoder unified --lrh --mfb \
      --lovasz_weight 1 --dice_weight 1 --output_dir outputs_ablation/FULL_seed$s
done

# ── leave-one-out 變體（adapter / loss 表）──
# A1 後置注入（= FULL 但 inject post），單 seed
run --seed 42 --inject post --decoder unified --lrh --mfb \
    --lovasz_weight 1 --dice_weight 1 --output_dir outputs_ablation/A1_seed42
# A2 移除 reference（= FULL 但 --no-ref），3 seeds
for s in $SEEDS_KEY; do
  run --seed $s --inject pre --decoder unified --lrh --mfb --no-ref \
      --lovasz_weight 1 --dice_weight 1 --output_dir outputs_ablation/A2_seed$s
done
# C1 純 CE（= FULL 但 loss=CE only），單 seed
run --seed 42 --inject pre --decoder unified --lrh --mfb \
    --lovasz_weight 0 --dice_weight 0 --output_dir outputs_ablation/C1_seed42

# ── 評估每個 run + 彙整 ──
for d in outputs_ablation/*/; do
  conda run -n sam_env python scripts/eval/eval_e1_acdc_val_full.py \
    --ckpt "$d/best.pth" --out "$d/e1_results.json" || echo "skip $d"
done
conda run -n sam_env python scripts/aggregate_ablation.py \
  --runs_root outputs_ablation --out outputs_ablation/ablation_tables.tex

echo "✅ 16 runs + eval + tables done."
```

> 確認 checkpoint 檔名（trainer 儲存的 best 權重檔名，可能非 `best.pth`）——讀 trainer 儲存邏輯後修正 `--ckpt` 路徑。

- [ ] **Step 2: 0-epoch smoke（驗證 10 config 都能跑通、config.json 落地）**

對每個 config 用 `--epochs 0`（或極小 epoch）跑一次，確認啟動、`ablation_config.json` 正確、eval 能據 config 重建。先跑 R1、FULL、A2 三個關鍵 config。
Run（範例）：
`conda run -n sam_env python segment-anything/train.py --epochs 0 --no-ref --output_dir /tmp/smoke_a2 && cat /tmp/smoke_a2/ablation_config.json`
Expected: config JSON 含 `"ref": false`。

- [ ] **Step 3: Commit**

```bash
git add segment-anything/run_ablation.sh
git commit -m "feat(ablation): add run_ablation.sh (16 runs + eval + aggregate)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11（可選 follow-up）: 遷移其餘 4 個非消融 LRH 呼叫點

> **與主線分離，不阻塞消融實驗。** 純一致性清理，消除未來分歧隱患。

**Files:**
- Modify: `scripts/eval/eval_e1_acdc_val_paper.py:80-86`、`scripts/eval/dump_acdc_test_preds.py:159-166`、`scripts/eval/viz_e4_qualitative.py:58-64`、`test_inference.py:55-62`

- [ ] **Step 1:** 各檔 import `assemble_semantic_logits`，把 `torch.full(...)` + 迴圈 + `context_fusion_head(full)` 改為 helper 呼叫（`use_lrh=getattr(model, 'use_lrh', True)`）。
- [ ] **Step 2:** Run 既有 eval/inference 各一次，確認輸出與遷移前一致（數值不變）。
- [ ] **Step 3: Commit**

```bash
git commit -am "refactor(ablation): migrate remaining LRH callers to shared helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 執行順序（依 spec §6）

1. **Task 1–9**（程式 + 測試，可連續完成；每 task 紅→綠→commit）。
2. **Task 10 Step 2 smoke**：先驗 R1/FULL/A2 三 config 跑通 + config 落地。
3. **訓練（算力主體）**：先 **FULL（3 seeds）** 作 pipeline sanity gate（val mIoU 須達 E27≈65.68 量級，否則先除錯）；再 A2(3)/R1(3)；再 R2–R6；再 A1/C1。
4. **eval + `aggregate_ablation.py`** → 3 張表 `.tex` + 正文數值。
5. **Task 11**（可選）。
6. 依 [`paper-rewrite-4.9-ablation.md`](../specs/2026-06-01-paper-rewrite-4.9-ablation.md) 改寫論文。

---

## Self-Review 註記

- **Spec 覆蓋**：5 開關（Task 3/4/5/7 + Task 2 lrh）、模組化 A（Task 1+2）、模組化 B（Task 6）、config 一致性（Task 7+8）、彙整（Task 9）、16 runs（Task 10）、可選 D（Task 11）皆有對應任務。
- **型別一致**：`assemble_semantic_logits(low_res_logits, class_ids, fusion_head, *, num_classes, use_lrh, fill_value)` 跨 Task 1/2/8/11 簽名一致；`model.use_lrh`、`mask_decoder.decoder_mode`、`vgg_injector.use_reference`、`ContextLoss(use_mfb=...)` 跨 task 命名一致。
- **執行期延展點**：僅 Task 9 表格組裝（依實測 JSON schema）與少數「實作前先讀上下文確認」標註；核心工具均有測試。
```
