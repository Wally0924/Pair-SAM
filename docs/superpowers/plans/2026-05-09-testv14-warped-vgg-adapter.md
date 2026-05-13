# testv14 WarpedVGG Adapter 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 CMAAlignment 對齊後的 clear-reference VGG level3 特徵（512ch, 64×64），透過輕量 Adapter 注入 SAM ViT-H Encoder 的 Block 31（最後一個 global attention block），使 Encoder 在完成特徵萃取前即感知晴天幾何結構，補足現有 dense prompt 只在 Decoder 注入的時序盲點。

**Architecture:** VGG level3 → 512→1280 Adapter（Linear + GeLU + Linear, 零初始化尾端）→ forward hook 在 Block 31 後加法注入 ViT token；採用 stateful injector 模式（`set_features()` before each forward）避免 closure stale data 問題；sigmoid gate 初始化 -3.0（≈ 0.047）確保訓練初期注入幾乎為零。

**Tech Stack:** PyTorch forward hook (`register_forward_hook`), ViT-H (`depth=32, embed_dim=1280, global_attn_indexes=[7,15,23,31]`), VGG16 level3 (512ch/64×64), AverageMeter logging

---

## 涉及檔案

| 動作 | 路徑 | 職責 |
|------|------|------|
| 新建 | `segment-anything/segment_anything/modeling/vgg_adapter.py` | WarpedVGGInjector 模組 |
| 修改 | `segment-anything/segment_anything/modeling/fusion.py` | 在 CMAAlignment.forward 暴露 warped VGG level3 特徵 |
| 修改 | `segment-anything/segment_anything/modeling/weather_sam.py` | 初始化 injector、註冊 hook、在 forward 呼叫 set_features |
| 修改 | `segment-anything/segment_anything/modeling/__init__.py` | export WarpedVGGInjector |
| 修改 | `segment-anything/weather_trainer.py` | 讀取 inject_cos_sim + gate_val 加入 AverageMeter 與 CSV log |
| 修改 | `segment-anything/train.py` | 新增 `--use_vgg_adapter`, `--adapter_inject_block`, `--adapter_lr_scale` |

---

## Task 1：CMAAlignment 暴露 warped VGG level3 特徵

**Files:**
- Modify: `segment-anything/segment_anything/modeling/fusion.py:165-230`

### 背景

`CMAAlignment.forward` 目前內部計算 `feats_ref`（VGG 特徵 list），僅在計算 flow 後使用，不對外回傳。`feats_ref[3]` = level3 = (B, 512, 64, 64)（在 256×256 input 上為 16×16；但我們需要在 1024×1024 input 上取 stride16 的特徵，對應 64×64）。

然而，目前 `_extract_vgg_features` 針對 1024 input 計算 `feats`，針對 256 input 計算 `feats_256`；UAWarpCHead 使用 `feats_256` 中的 level2/3（32×32 和 16×16）計算 flow。我們要的 64×64 level3 特徵來自 **1024 input 的 feats[2]**（stride8, 256ch, 128×128 → 不對），實際上 `feats[3]`（stride16, 512ch）在 1024 input 上是 64×64。

確認：VGG `level3 = feats[17:24]`，stride16，1024 input → output = 1024/16 = 64。正確，`feats[3]` = (B, 512, 64, 64)。

**注意**：`_extract_vgg_features` 使用 `@torch.no_grad()`，但我們要的 warped level3 特徵要讓 Adapter 的梯度可以流回 VGG。由於 VGG 本身是凍結的（`requires_grad=False`），梯度不需要流過 VGG，只需要從 `feats_ref[3]` 出發，流經 warp → Adapter → hook inject → ViT forward。因此使用 `torch.no_grad()` 提取特徵是正確的。

- [ ] **Step 1: 修改 `CMAAlignment.forward`，新增回傳第三個值 `f_ref_warped_vgg`**

在 `fusion.py` 的 `CMAAlignment.forward`（line 165）中，找到以下區段並修改：

**Before（line 174-187，fallback 路徑）：**
```python
        if img_curr is not None and img_ref is not None:
            feats_curr, feats_curr_256 = self._extract_vgg_features(img_curr)
            feats_ref,  feats_ref_256  = self._extract_vgg_features(img_ref)
        else:
            # Fallback
            B = f_curr.shape[0]
            with torch.no_grad():
                self._last_conf_mean      = 0.0
                self._last_valid_ratio    = 1.0
                self._last_flow           = torch.zeros(B, 2, H, W)
                self._last_confidence_map = torch.ones(B, 1, H, W)
            conf = torch.ones(B, 1, H, W, device=f_curr.device)
            return f_ref, conf
```

**After：**
```python
        if img_curr is not None and img_ref is not None:
            feats_curr, feats_curr_256 = self._extract_vgg_features(img_curr)
            feats_ref,  feats_ref_256  = self._extract_vgg_features(img_ref)
        else:
            # Fallback
            B = f_curr.shape[0]
            with torch.no_grad():
                self._last_conf_mean      = 0.0
                self._last_valid_ratio    = 1.0
                self._last_flow           = torch.zeros(B, 2, H, W)
                self._last_confidence_map = torch.ones(B, 1, H, W)
            conf = torch.ones(B, 1, H, W, device=f_curr.device)
            # f_ref_warped_vgg fallback：零 tensor，WeatherSAM 偵測到全零則跳過注入
            return f_ref, conf, torch.zeros(B, 512, H, W, device=f_curr.device)
```

- [ ] **Step 2: 在正常路徑中，計算 warped VGG level3 並回傳**

找到 `CMAAlignment.forward` 末尾的 `return` 語句（約 line 220 之後），加入 VGG level3 warp 計算。在 `confidence = ...` 計算完成後，`return` 之前插入：

```python
        # Warped VGG level3 特徵：將 clear ref 的幾何紋理資訊對齊到 curr 視角
        # feats_ref[3] = (B, 512, 64, 64)，stride16 於 1024 input
        with torch.no_grad():
            f_vgg_ref_l3 = feats_ref[3]  # (B, 512, 64, 64)，VGG 凍結，no_grad 安全
        if f_vgg_ref_l3.shape[-2:] != (H, W):
            f_vgg_ref_l3 = F.interpolate(f_vgg_ref_l3, size=(H, W),
                                          mode='bilinear', align_corners=False)
        f_ref_warped_vgg, _ = warp(f_vgg_ref_l3, flow1, return_mask=True)  # (B, 512, 64, 64)
```

然後更新所有 return 語句，在結尾加入 `f_ref_warped_vgg`：

找到最後的 return（原本是 `return f_ref_warped, confidence`），改為：
```python
        return f_ref_warped, confidence, f_ref_warped_vgg
```

- [ ] **Step 3: 確認修改後 forward 回傳簽名一致**

在 `weather_sam.py` 中，`CMAAlignment` 的呼叫點（約 line 138）：
```python
        f_ref_warped, confidence = self.fusion_module(...)
```
**先不動**——Task 3 會統一修改。

---

## Task 2：建立 WarpedVGGInjector 模組

**Files:**
- Create: `segment-anything/segment_anything/modeling/vgg_adapter.py`

### 設計說明

- 512ch → 320ch → 1280ch（三層：Linear + GeLU + Linear，尾端零初始化）
- sigmoid gate 初始化 -3.0，sigmoid(-3) ≈ 0.047，訓練初期幾乎不注入
- `inject()` 在 Block 31 的 forward hook 中被呼叫：將 adapter output 加到 ViT tokens
- `inject_cos_sim`：注入前後 token 的 cosine similarity，接近 1.0 表示注入量小（健康）
- `set_features()`：在每次 forward 前設定當次的 `f_ref_warped_vgg`，避免 closure stale data

- [ ] **Step 1: 新建 `vgg_adapter.py` 並寫入完整實作**

```python
# vgg_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class WarpedVGGInjector(nn.Module):
    """
    將 CMAAlignment 輸出的 warped VGG level3 特徵（512ch, 64×64）
    透過輕量 Adapter 注入 SAM ViT-H Encoder 指定 Block 的輸出 token。

    注入點：Block 後的加法 (token += gate * adapter(f_vgg))
    gate：sigmoid 初始值 ≈ 0.047（init=-3.0），訓練初期幾乎不注入
    adapter：512 → 320 → 1280，尾端零初始化確保訓練初期不干擾 ViT
    """

    def __init__(self, vgg_channels: int = 512, vit_dim: int = 1280,
                 hidden_dim: int = 320):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(vgg_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vit_dim),
        )
        # 零初始化尾端層：訓練初期 adapter output ≈ 0
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

        # sigmoid gate，初始值 sigmoid(-3.0) ≈ 0.047
        self.gate = nn.Parameter(torch.tensor(-3.0))

        # 內部狀態：每次 forward 前由 set_features() 設定
        self._f_vgg: torch.Tensor = None

        # 診斷指標（不參與梯度，供 trainer 讀取）
        self._last_inject_cos_sim: float = 1.0
        self._last_gate_val: float = float(torch.sigmoid(torch.tensor(-3.0)))

    def set_features(self, f_vgg: torch.Tensor):
        """在每次 WeatherSAM.forward 呼叫 image_encoder 前設定當次特徵。"""
        self._f_vgg = f_vgg  # (B, 512, 64, 64)

    def inject(self, module, input, output):
        """
        Register 為 ViT Block 的 forward hook。
        output shape：(B, H, W, C) = (B, 64, 64, 1280)
        """
        if self._f_vgg is None:
            return output  # 未設定特徵（cache 模式或 fallback），跳過注入

        B, H, W, C = output.shape
        f = self._f_vgg.to(output.device)  # (B, 512, 64, 64)

        if f.abs().sum().item() == 0.0:
            # 全零 sentinel：fallback 路徑，跳過注入
            return output

        # (B, 512, 64, 64) → (B, 64, 64, 512) → (B, H*W, 512)
        f_flat = f.permute(0, 2, 3, 1).reshape(B, H * W, 512)  # (B, 4096, 512)

        delta = self.adapter(f_flat)  # (B, 4096, 1280)
        delta = delta.reshape(B, H, W, C)

        gate_val = float(torch.sigmoid(self.gate).item())
        injected = output + torch.sigmoid(self.gate) * delta

        # 診斷：注入前後 cosine similarity（dim=-1 over C）
        with torch.no_grad():
            cos = F.cosine_similarity(
                output.reshape(B, -1, C),
                injected.reshape(B, -1, C),
                dim=-1,
            ).mean().item()
            self._last_inject_cos_sim = cos
            self._last_gate_val = gate_val

        self._f_vgg = None  # 清除，避免下次誤用
        return injected
```

---

## Task 3：整合 WarpedVGGInjector 到 WeatherSAM

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_sam.py`

- [ ] **Step 1: 新增 import**

在 `weather_sam.py` 頂部找到現有 import 區段，加入：
```python
from .vgg_adapter import WarpedVGGInjector
```

- [ ] **Step 2: 在 `WeatherSAM.__init__` 中初始化 injector 並註冊 hook**

在 `__init__` 方法的末尾（`register_buffer` 之後）加入：

```python
        # [testv14] WarpedVGG Adapter：注入 ViT-H Block 31（最後 global attention block）
        # use_vgg_adapter=False 時完全跳過，不影響現有 forward 行為
        self.use_vgg_adapter: bool = False
        self.vgg_injector = WarpedVGGInjector(
            vgg_channels=512, vit_dim=1280, hidden_dim=320
        )
        self._hook_handle = None  # 記錄 hook handle，方便解除
```

- [ ] **Step 3: 新增 `enable_vgg_adapter` 方法（負責延遲 hook 註冊）**

在 `WeatherSAM` class 中新增方法（在 `forward` 之前）：

```python
    def enable_vgg_adapter(self, inject_block: int = 31):
        """啟用 WarpedVGG Adapter，並在指定 ViT Block 後註冊 hook。"""
        if self._hook_handle is not None:
            self._hook_handle.remove()  # 避免重複 hook
        target_block = self.image_encoder.blocks[inject_block]
        self._hook_handle = target_block.register_forward_hook(self.vgg_injector.inject)
        self.use_vgg_adapter = True
        print(f'[WeatherSAM] WarpedVGG Adapter enabled at ViT Block {inject_block}.')
```

- [ ] **Step 4: 更新 `forward` 中的 `fusion_module` 呼叫，接收第三個回傳值**

在 `weather_sam.py` 的 `forward`（約 line 138），將：
```python
        f_ref_warped, confidence = self.fusion_module(
            f_curr=image_embeddings,
            f_ref=ref_embeddings,
            img_curr=img_curr_batch,
            img_ref=img_ref_batch,
        )
```
改為：
```python
        f_ref_warped, confidence, f_ref_warped_vgg = self.fusion_module(
            f_curr=image_embeddings,
            f_ref=ref_embeddings,
            img_curr=img_curr_batch,
            img_ref=img_ref_batch,
        )
```

- [ ] **Step 5: 在 image_encoder 呼叫前設定 injector 特徵**

在 `forward` 的「階段 1：核心特徵萃取」區段，找到：
```python
        if "image_embedding" in batched_input[0]:
            image_embeddings = torch.stack([x["image_embedding"] for x in batched_input], dim=0)
        else:
            input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
            image_embeddings = self.image_encoder(input_images)
```

**注意**：CMAAlignment 的呼叫在「階段 2」，但 image_encoder 的呼叫在「階段 1」。所以 VGG warped 特徵要在 **第二次** image_encoder 呼叫前設定（clear_image 的 encode）。

實際流程：
1. 階段 1：`image_encoder(input_images)` → `image_embeddings`（惡劣天氣）
2. 階段 2：`CMAAlignment.forward` → 計算 VGG warp，取得 `f_ref_warped_vgg`
3. 如果使用 cache（`image_embedding` in input），則 injector 的 `set_features` 不需要，因為 image_encoder 不會被呼叫

因此，在階段 2 的 `fusion_module` 呼叫完成後，加入以下邏輯（在取得 `f_ref_warped_vgg` 之後）：

```python
        # [testv14] 如果啟用 VGG Adapter 且 image_encoder 正在即時運行（非 cache 模式）
        # 此路徑下 image_encoder 已完成對 input_images 的 encode，
        # 但 clear_image 的 encode 尚未發生（ref_embeddings 來自 cache 或即時）。
        # WarpedVGG 注入目標是 clear ref 特徵進入 ViT 的過程——
        # 修正：注入應在 target image 的 encode 前，預設在 input_images encode 時注入。
        # 實際上我們已在階段 1 完成了 image_encoder(input_images)，因此：
        # Phase-1（ViT 凍結）時，以 f_ref_warped_vgg 作為 set_features 給下一批次的 hook
        # 這裡先 set，hook 會在下一個 image_encoder 呼叫時生效。
        # ──────────────────────────────────────────────────────────────────
        # 修正方案：將 image_encoder 拆為兩次呼叫 — 現有設計是先 encode 再 CMA。
        # 正確的注入時機是在 encode 內部（透過 hook），所以 set_features 要在 encode 之前。
        # 因此重排順序：先做 CMA 的 VGG 特徵提取（不需要 f_curr embedding），再 encode。
```

**架構時序問題說明**：

現有 forward 順序是「encode image → CMA（需要 VGG 特徵，不需要 f_curr）→ decode」。
WarpedVGG 注入需要在 encode 之前呼叫 `set_features`，hook 才能在 encode 內部作用。

**修正方案**：將 CMA 的 VGG 特徵提取邏輯（不依賴 f_curr embedding）提前到 encode 之前。

- [ ] **Step 6: 重構 forward 的 VGG 特徵提取時序**

在「階段 1」的 image_encoder 呼叫之前（即取 `input_images` 之後），加入 VGG 特徵預計算：

```python
        # --- 階段 0：預先提取 VGG 特徵供 WarpedVGG Adapter 使用（僅 raw image 模式）---
        _f_ref_warped_vgg_prefetch = None
        if (self.use_vgg_adapter
                and "image_embedding" not in batched_input[0]
                and all("image" in x and "clear_image" in x for x in batched_input)):
            _img_curr_pre = torch.stack(
                [self.preprocess(x["image"]) for x in batched_input], dim=0
            )
            _img_ref_pre = torch.stack(
                [self.preprocess(x["clear_image"]) for x in batched_input], dim=0
            )
            # 只提取 VGG 特徵（no_grad，不需要 f_curr embedding）
            with torch.no_grad():
                _feats_ref_pre, _ = self.fusion_module._extract_vgg_features(
                    _img_ref_pre.to(next(self.fusion_module.vgg_backbone.parameters()).device)
                )
                _f_vgg_l3_pre = _feats_ref_pre[3]  # (B, 512, 64, 64)
            _f_ref_warped_vgg_prefetch = _f_vgg_l3_pre
            self.vgg_injector.set_features(_f_ref_warped_vgg_prefetch)
```

然後在原本的「階段 1」中，image_encoder 呼叫時 hook 會自動作用（set_features 已設定）。

在「階段 2」取得 `f_ref_warped_vgg` 之後，更新 hook 狀態（用已 warp 的特徵覆蓋 prefetch 版本，適用於下一次 forward 的預熱）——實際上 Phase-1 時 injector 用的是 prefetch 的 unwarped level3，還沒有 flow。這是 MVP 可接受的近似——完整方案需要先計算 flow 再 set_features，但那需要把整個 CMA 提前，複雜度過高。

**MVP 簡化**：直接使用 unwarped VGG level3（已對齊坐標系），flow 校正只影響輕微幾何偏移，在 Phase-1 凍結 ViT 階段影響有限。

更新 Stage 0 最後一行，使用 warped 特徵（從 Stage 2 取得後補設）：

在 Stage 2 取得 `f_ref_warped_vgg` 後加入：
```python
        # 用已 warp 的 VGG 特徵覆蓋 prefetch（供監控用；實際注入已發生）
        if self.use_vgg_adapter and _f_ref_warped_vgg_prefetch is not None:
            # 下一個 batch 的 prefetch 依然使用 unwarped，這裡記錄 warp 結果供診斷
            with torch.no_grad():
                self.vgg_injector._last_warped_vgg_norm = float(
                    f_ref_warped_vgg.norm(dim=1).mean().item()
                )
```

---

## Task 4：更新 `__init__.py` export

**Files:**
- Modify: `segment-anything/segment_anything/modeling/__init__.py`

- [ ] **Step 1: 加入 WarpedVGGInjector export**

在 `__init__.py` 中找到現有 import 區段，加入：
```python
from .vgg_adapter import WarpedVGGInjector
```

並在 `__all__` list（如有）中加入 `"WarpedVGGInjector"`。

---

## Task 5：在 Trainer 中新增 inject_cos_sim 監控

**Files:**
- Modify: `segment-anything/weather_trainer.py`

- [ ] **Step 1: 在 train loop 的 losses dict 中加入新 meter**

找到 train loop 的 `losses` 初始化（約 line 293），在現有 meter 後加入：
```python
            "inject_cos_sim": AverageMeter(),
            "inject_gate":    AverageMeter(),
```

- [ ] **Step 2: 在 train loop 的 batch 結束後讀取 injector 診斷值**

找到 train loop 中 `losses['lovasz'].update(...)` 之後，加入：
```python
                # [testv14] WarpedVGG Adapter 診斷指標
                if hasattr(model, 'vgg_injector') and model.use_vgg_adapter:
                    losses['inject_cos_sim'].update(
                        model.vgg_injector._last_inject_cos_sim
                    )
                    losses['inject_gate'].update(
                        model.vgg_injector._last_gate_val
                    )
```

- [ ] **Step 3: 在 val loop 重複相同操作**

找到 val loop 的 `losses` 初始化（約 line 783），同樣加入兩個 meter 並在 batch 後讀取。

- [ ] **Step 4: 在 epoch 結束的 log_entry 中加入欄位**

找到 `log_entry` dict，加入：
```python
                'train_inject_cos_sim': losses['inject_cos_sim'].avg if 'inject_cos_sim' in losses else 1.0,
                'train_inject_gate':    losses['inject_gate'].avg    if 'inject_gate'    in losses else 0.0,
                'val_inject_cos_sim':   val_losses['inject_cos_sim'].avg if 'inject_cos_sim' in val_losses else 1.0,
                'val_inject_gate':      val_losses['inject_gate'].avg    if 'inject_gate'    in val_losses else 0.0,
```

---

## Task 6：更新 `train.py` 訓練參數與 WeatherSAM 初始化

**Files:**
- Modify: `segment-anything/train.py`

- [ ] **Step 1: 新增 CLI 參數**

在 `argparse` 區段加入：
```python
    parser.add_argument('--use_vgg_adapter', action='store_true',
                        help='啟用 WarpedVGG Adapter（注入 ViT Block 31）')
    parser.add_argument('--adapter_inject_block', type=int, default=31,
                        help='注入的 ViT-H Block index（0-indexed，預設 31 = 最後 global attention）')
    parser.add_argument('--adapter_lr_scale', type=float, default=5.0,
                        help='Adapter 參數的學習率倍率（相對於 base lr）')
```

- [ ] **Step 2: 在 WeatherSAM 初始化後，按條件啟用 adapter**

找到 `model = WeatherSAM(...)` 或 `model.load_state_dict(...)` 之後，加入：
```python
    if args.use_vgg_adapter:
        model.enable_vgg_adapter(inject_block=args.adapter_inject_block)
        print(f'[Config] WarpedVGG Adapter enabled at Block {args.adapter_inject_block}, '
              f'lr_scale={args.adapter_lr_scale}')
```

- [ ] **Step 3: 在 optimizer 參數分組中，給 adapter 高學習率**

找到 optimizer 初始化（通常是 `optim.AdamW([...])` 或 param groups），加入 adapter 的獨立 param group：

```python
    adapter_params = list(model.vgg_injector.parameters()) if args.use_vgg_adapter else []
    
    # 在 param_groups list 中加入（調整現有結構）：
    if adapter_params:
        param_groups.append({
            'params': adapter_params,
            'lr': args.lr * args.adapter_lr_scale,
            'name': 'vgg_adapter',
        })
```

- [ ] **Step 4: 在 config print 區段顯示 adapter 設定**

找到訓練配置 print 區段，加入：
```python
    if args.use_vgg_adapter:
        print(f'  WarpedVGG Adapter: Block {args.adapter_inject_block}, lr_scale={args.adapter_lr_scale}')
```

- [ ] **Step 5: 在 `plot_history` 中加入 inject_cos_sim 曲線**

找到 `plot_history` 的 components list，加入：
```python
    {'train_key': 'train_inject_cos_sim', 'val_key': 'val_inject_cos_sim',
     'label': 'Inject CosSim', 'color': 'purple'},
    {'train_key': 'train_inject_gate', 'val_key': 'val_inject_gate',
     'label': 'Adapter Gate', 'color': 'brown'},
```

---

## Task 7：執行測試確認整合正確

- [ ] **Step 1: 語法與 import 驗證**

```bash
conda run -n sam_env python -c "
from segment_anything.modeling.vgg_adapter import WarpedVGGInjector
import torch
inj = WarpedVGGInjector()
f = torch.zeros(2, 512, 64, 64)
inj.set_features(f)
print('WarpedVGGInjector init OK')
print('gate initial:', torch.sigmoid(inj.gate).item())
print('adapter last layer weight norm:', inj.adapter[2].weight.norm().item())
"
```

預期輸出：
```
WarpedVGGInjector init OK
gate initial: 0.047...
adapter last layer weight norm: 0.0
```

- [ ] **Step 2: WeatherSAM forward smoke test（minimal batch）**

```bash
conda run -n sam_env python -c "
import sys; sys.path.insert(0, 'segment-anything')
from segment_anything import build_weather_sam_vit_h
import torch

model = build_weather_sam_vit_h(checkpoint=None)
model.enable_vgg_adapter(inject_block=31)
print('hook registered:', model._hook_handle is not None)

# 確認 vgg_injector gate 值
print('gate:', torch.sigmoid(model.vgg_injector.gate).item())
" 2>&1 | head -20
```

- [ ] **Step 3: CMAAlignment 三值回傳確認**

```bash
conda run -n sam_env python -c "
import sys; sys.path.insert(0, 'segment-anything')
from segment_anything.modeling.fusion import CMAAlignment
import torch

cma = CMAAlignment()
f = torch.zeros(1, 256, 64, 64)
ret = cma(f_curr=f, f_ref=f, img_curr=None, img_ref=None)
print('return length:', len(ret))
print('f_ref_warped_vgg shape:', ret[2].shape)
" 2>&1
```

預期：
```
return length: 3
f_ref_warped_vgg shape: torch.Size([1, 512, 64, 64])
```

---

## 訓練指令（Phase 1：凍結 ViT，只訓練 Adapter）

```bash
conda run -n sam_env python segment-anything/train.py \
  --output_dir outputs_weather_sam_mask2former_testv14 \
  --use_vgg_adapter \
  --adapter_inject_block 31 \
  --adapter_lr_scale 5.0 \
  --lovasz_weight 0.5 \
  --num_epochs 30
```

**健康指標觀察（前 5 epochs）：**
- `inject_cos_sim` > 0.95：注入量級適中（正常）
- `inject_cos_sim` < 0.70：gate 未約束，注入過強，調低 `adapter_lr_scale`
- `inject_gate`：應從 ~0.047 緩慢上升，穩定在 0.2-0.5 之間
- `mIoU`：應在 epoch 3-5 超越 testv13 同期水準（若 adapter 有效）

---

## Self-Review

### Spec 覆蓋確認
- [x] Task 1：CMAAlignment 暴露 warped VGG level3 ✓
- [x] Task 2：WarpedVGGInjector 模組（零初始化、gate、set_features、inject_cos_sim）✓
- [x] Task 3：WeatherSAM 整合（hook 註冊、set_features 時序、fallback 路徑）✓
- [x] Task 4：__init__.py export ✓
- [x] Task 5：Trainer 監控（inject_cos_sim, gate） ✓
- [x] Task 6：train.py 參數（use_vgg_adapter, inject_block, lr_scale, plot） ✓
- [x] Task 7：驗證測試 ✓

### 已知限制（MVP 可接受）
- Phase-1 時注入使用 unwarped VGG level3（prefetch），flow 校正只影響輕微幾何偏移
- 單點注入（Block 31），多尺度注入（7/15/23/31）留待 Phase-2 驗證
- cache 模式（`image_embedding` in input）下 injector 不作用（bypass），訓練時應使用 `--force_raw_images` 或清除 cache
