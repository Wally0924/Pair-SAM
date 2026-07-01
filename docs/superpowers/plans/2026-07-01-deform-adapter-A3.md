# WeatherSAM 雙向可變形 Adapter(A3)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把單向的 `MultiScaleCrossAttnInjector` 換成忠於 ViT-Adapter 的雙向多尺度可變形 adapter(Injector + Extractor),Spatial Prior Module 由 UAWarpC 參考對齊擔任;僅動 encoder 端,decoder/LRH/輸出不變。

**Architecture:** RPM 從 UAWarpC 對齊的 VGG 參考建 3 尺度 token 流 `c`;ViT-H 32 層切 4 組,pre-hook@0/8/16/24 注入(Q=ViT.detach(), K/V=c, MSDeformAttn n_levels=3),post-hook@7/15/23 抽取(Q=c, K/V=ViT.detach(), n_levels=1, +ConvFFN)更新 `c`;末組只注入。梯度雙向都不改 ViT。

**Tech Stack:** PyTorch 2.9.1+cu128、pure-PyTorch MSDeformAttn(免編譯)、pytest 9、conda env `sam_env`。

## Global Constraints

- 執行任何 python/pytest:`conda run -n sam_env python ...`(exact)。
- 非侵入:不改 `image_encoder.py` 的 block 結構/權重,只用 forward hook。
- 單尺度輸出:不動 `weather_mask_decoder.py` / `fusion_head.py`(LRH)/ 輸出組裝。
- ViT 保護:Injector 對 query `.detach()`;Extractor 對 ViT feat `.detach()`;殘差路徑保留梯度。
- Gate:`softplus(raw)`,init raw = `math.log(math.exp(0.05)-1)`(≈ softplus 0.05),無上界,沿用 trainer warmup。
- 注入尺度來源:l2=1/8(256ch)、l3=1/16(512ch)來自 `fusion.pre_align`;1/32 由 l3 stride-2 conv 降採。
- confidence:value 投影前乘 per-token 信心(僅 injector);extractor 不加信心。
- `deform_ratio=0.5`。`n_points=4`。interaction dim = `vit_dim`(ViT-H=1280)。
- Ablation 相容:`use_vgg_adapter=False`(不掛 hook)、`use_reference=False`(RPM 零化 c)。
- 測試置於 `segment-anything/tests/`,沿用 `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))` 慣例、小維度、CPU 可跑。

---

### Task 1: Vendor pure-PyTorch MSDeformAttn(里程碑 0 去風險)

**Files:**
- Create: `segment-anything/segment_anything/modeling/ops/__init__.py`
- Create: `segment-anything/segment_anything/modeling/ops/ms_deform_attn.py`
- Test: `segment-anything/tests/test_ms_deform_attn.py`

**Interfaces:**
- Produces:
  - `ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights) -> Tensor`
  - `class MSDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0)`,
    `forward(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None) -> (N, Len_q, d_model)`,
    method `_reset_parameters()`。

- [ ] **Step 1: 寫失敗測試**

```python
# segment-anything/tests/test_ms_deform_attn.py
"""pure-PyTorch MSDeformAttn 去風險測試（無 CUDA 編譯）。
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_ms_deform_attn.py -v
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.ops.ms_deform_attn import MSDeformAttn


def _ref_points(spatial_shapes, device):
    refs = []
    for (H, W) in spatial_shapes:
        ry, rx = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device) / H,
            torch.linspace(0.5, W - 0.5, W, device=device) / W,
            indexing='ij')
        refs.append(torch.stack((rx.reshape(-1), ry.reshape(-1)), -1))
    return torch.cat(refs, 0)[None, :, None, :]  # (1, Lq, 1, 2)


def test_forward_shape_preserved():
    d, N = 32, 2
    shapes = [(8, 8), (4, 4)]
    lens = [h * w for h, w in shapes]
    lsi = torch.tensor([0, lens[0]])
    attn = MSDeformAttn(d_model=d, n_levels=len(shapes), n_heads=4, n_points=4, ratio=0.5)
    value = torch.randn(N, sum(lens), d)
    query = torch.randn(N, lens[0], d)  # query on level-0 grid
    ref = _ref_points([shapes[0]], value.device).expand(N, -1, -1, -1)
    out = attn(query, ref, value, torch.tensor(shapes), lsi)
    assert out.shape == (N, lens[0], d)
    assert torch.isfinite(out).all()


def test_gradient_flows_to_value_and_params():
    d, N = 32, 1
    shapes = [(8, 8)]
    lens = [h * w for h, w in shapes]
    attn = MSDeformAttn(d_model=d, n_levels=1, n_heads=4, n_points=4, ratio=0.5)
    value = torch.randn(N, sum(lens), d, requires_grad=True)
    query = torch.randn(N, lens[0], d)
    ref = _ref_points(shapes, value.device).expand(N, -1, -1, -1)
    attn(query, ref, value, torch.tensor(shapes), torch.tensor([0])).sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert attn.sampling_offsets.weight.grad is not None
```

- [ ] **Step 2: 執行確認失敗**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_ms_deform_attn.py -v`
Expected: FAIL — `ModuleNotFoundError: segment_anything.modeling.ops`

- [ ] **Step 3: 建 ops 套件與 MSDeformAttn(完整 vendored 程式碼)**

```python
# segment-anything/segment_anything/modeling/ops/__init__.py
from .ms_deform_attn import MSDeformAttn, ms_deform_attn_core_pytorch
```

```python
# segment-anything/segment_anything/modeling/ops/ms_deform_attn.py
"""Pure-PyTorch Multi-Scale Deformable Attention（免 CUDA 編譯）。
移植自 Deformable-DETR / ViT-Adapter 的 ms_deform_attn_core_pytorch，
以 F.grid_sample 實作，torch 2.x 相容。CUDA 加速版為可選後續優化，非本計畫關鍵路徑。
"""
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


def _is_power_of_2(n):
    if (not isinstance(n, int)) or n < 0:
        raise ValueError(f"invalid input for _is_power_of_2: {n}")
    return (n & (n - 1) == 0) and n != 0


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    value:               (N, S, n_heads, head_dim)
    value_spatial_shapes:(n_levels, 2)
    sampling_locations:  (N, Lq, n_heads, n_levels, n_points, 2)  in [0,1]
    attention_weights:   (N, Lq, n_heads, n_levels, n_points)
    return:              (N, Lq, n_heads*head_dim)
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape
    value_list = value.split([int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
              ).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("MSDeformAttn: d_model/n_heads 非 2 的次方，CUDA 版效率較差（本 pure-torch 版不受影響）。")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        n_head_dim = int(self.d_model * self.ratio) // self.n_heads
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, n_head_dim)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = attention_weights.softmax(-1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).to(query.device)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)
```

- [ ] **Step 4: 執行確認通過**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_ms_deform_attn.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/ops/ segment-anything/tests/test_ms_deform_attn.py
git commit -m "feat(adapter): vendor pure-PyTorch MSDeformAttn (A3 milestone 0)"
```

---

### Task 2: deform_inputs 幾何輔助

**Files:**
- Create: `segment-anything/segment_anything/modeling/deform_adapter.py`(先只放輔助函式)
- Test: `segment-anything/tests/test_deform_inputs.py`

**Interfaces:**
- Produces:
  - `get_reference_points(spatial_shapes: list[tuple], device) -> Tensor (1, sum_L, 1, 2)`
  - `deform_inputs(h: int, w: int, device) -> (inject_inputs, extract_inputs)`,各為
    `[reference_points, spatial_shapes(LongTensor (L,2)), level_start_index(LongTensor (L,))]`。
    inject:query=1/16 grid,value=3 尺度(1/8,1/16,1/32)。
    extract:query=3 尺度,value=1/16 單尺度。

- [ ] **Step 1: 寫失敗測試**

```python
# segment-anything/tests/test_deform_inputs.py
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import get_reference_points, deform_inputs


def test_reference_points_shape_and_range():
    ref = get_reference_points([(4, 4), (2, 2)], torch.device('cpu'))
    assert ref.shape == (1, 16 + 4, 1, 2)
    assert ref.min() >= 0.0 and ref.max() <= 1.0


def test_deform_inputs_inject_and_extract():
    h = w = 64  # ViT token grid = 1/16 of 1024
    inj, ext = deform_inputs(h, w, torch.device('cpu'))
    # inject: value 有 3 尺度 (h*2)²,(h)²,(h/2)²
    assert inj[1].tolist() == [[128, 128], [64, 64], [32, 32]]
    assert inj[0].shape == (1, 64 * 64, 1, 2)          # query = 1/16 grid
    assert inj[2].tolist() == [0, 128 * 128, 128 * 128 + 64 * 64]
    # extract: value 為單一 1/16，query = 3 尺度
    assert ext[1].tolist() == [[64, 64]]
    assert ext[0].shape == (1, 128 * 128 + 64 * 64 + 32 * 32, 1, 2)
    assert ext[2].tolist() == [0]
```

- [ ] **Step 2: 執行確認失敗**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_inputs.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_reference_points'`

- [ ] **Step 3: 實作輔助函式**

```python
# segment-anything/segment_anything/modeling/deform_adapter.py
"""WeatherSAM 雙向可變形 Adapter（A3）。SPM → UAWarpC 參考；Injector + Extractor。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops.ms_deform_attn import MSDeformAttn

_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # softplus(x) ≈ 0.05


def get_reference_points(spatial_shapes, device):
    refs = []
    for (H_, W_) in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing='ij')
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        refs.append(torch.stack((ref_x, ref_y), -1))
    reference_points = torch.cat(refs, 1)[:, :, None]  # (1, sum_L, 1, 2)
    return reference_points


def deform_inputs(h, w, device):
    """h,w = ViT token grid（1/16 of input）。value 三尺度 = 1/8,1/16,1/32。"""
    c_shapes = torch.as_tensor([(h * 2, w * 2), (h, w), (h // 2, w // 2)],
                               dtype=torch.long, device=device)
    c_lsi = torch.cat((c_shapes.new_zeros((1,)), c_shapes.prod(1).cumsum(0)[:-1]))
    inject = [get_reference_points([(h, w)], device), c_shapes, c_lsi]

    vit_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=device)
    vit_lsi = torch.cat((vit_shapes.new_zeros((1,)), vit_shapes.prod(1).cumsum(0)[:-1]))
    extract = [get_reference_points([(h * 2, w * 2), (h, w), (h // 2, w // 2)], device),
               vit_shapes, vit_lsi]
    return inject, extract
```

- [ ] **Step 4: 執行確認通過**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_inputs.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/deform_adapter.py segment-anything/tests/test_deform_inputs.py
git commit -m "feat(adapter): add deform_inputs geometry helpers (A3)"
```

---

### Task 3: ReferencePriorModule(RPM)

**Files:**
- Modify: `segment-anything/segment_anything/modeling/deform_adapter.py`(append `ReferencePriorModule`)
- Test: `segment-anything/tests/test_rpm.py`

**Interfaces:**
- Consumes:`{'l2':(B,256,H8,W8),'l3':(B,512,H16,W16),'mask':(B,1,Hm,Wm) 可選}`。
- Produces:`class ReferencePriorModule(l2_channels=256, l3_channels=512, dim=1280, use_reference=True)`,
  `forward(feats) -> (c, conf)`;`c`:(B, L, dim) 三尺度串接 + level_embed;
  `conf`:(B, L, 1) per-token 信心(無 mask 時全 1)。`use_reference=False` → `c` 零化(仍回原形狀)。

- [ ] **Step 1: 寫失敗測試**

```python
# segment-anything/tests/test_rpm.py
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import ReferencePriorModule


def _feats(B=1, H16=4):
    H8 = H16 * 2
    return {'l2': torch.randn(B, 8, H8, H8),
            'l3': torch.randn(B, 16, H16, H16),
            'mask': torch.rand(B, 1, H8, H8)}


def test_c_shape_three_scales_concat():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32)
    c, conf = rpm(_feats(H16=4))
    L = 8 * 8 + 4 * 4 + 2 * 2           # 1/8 + 1/16 + 1/32
    assert c.shape == (1, L, 32)
    assert conf.shape == (1, L, 1)


def test_level_embed_makes_scales_distinguishable():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32)
    assert rpm.level_embed.shape == (3, 32)


def test_use_reference_false_zeros_c_keeps_shape():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32, use_reference=False)
    c, conf = rpm(_feats(H16=4))
    L = 8 * 8 + 4 * 4 + 2 * 2
    assert c.shape == (1, L, 32)
    assert c.abs().max().item() == 0.0


def test_conf_all_ones_without_mask():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32)
    feats = _feats(H16=4); del feats['mask']
    c, conf = rpm(feats)
    assert torch.allclose(conf, torch.ones_like(conf))
```

- [ ] **Step 2: 執行確認失敗**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_rpm.py -v`
Expected: FAIL — `ImportError: cannot import name 'ReferencePriorModule'`

- [ ] **Step 3: 實作 RPM(append 至 deform_adapter.py)**

```python
class ReferencePriorModule(nn.Module):
    """取代 ViT-Adapter SPM：把 UAWarpC 對齊的 VGG 參考轉成 3 尺度 token 流。
    1/8 ← l2；1/16 ← l3；1/32 ← l3 stride-2 降採（決策②）。"""
    def __init__(self, l2_channels=256, l3_channels=512, dim=1280, use_reference=True):
        super().__init__()
        self.dim = dim
        self.use_reference = use_reference
        self.proj_c2 = nn.Conv2d(l2_channels, dim, kernel_size=1)
        self.proj_c3 = nn.Conv2d(l3_channels, dim, kernel_size=1)
        self.down_c4 = nn.Conv2d(l3_channels, dim, kernel_size=3, stride=2, padding=1)
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(self, feats):
        l2 = feats['l2']; l3 = feats['l3']
        B = l2.shape[0]
        c2 = self.proj_c2(l2)                    # (B,dim,H8,W8)
        c3 = self.proj_c3(l3)                    # (B,dim,H16,W16)
        c4 = self.down_c4(l3)                    # (B,dim,H32,W32)

        def _flat(x):
            return x.flatten(2).transpose(1, 2)  # (B, H*W, dim)
        t2, t3, t4 = _flat(c2), _flat(c3), _flat(c4)
        t2 = t2 + self.level_embed[0]
        t3 = t3 + self.level_embed[1]
        t4 = t4 + self.level_embed[2]
        c = torch.cat([t2, t3, t4], dim=1)       # (B, L, dim)

        if not self.use_reference:
            c = torch.zeros_like(c)

        mask = feats.get('mask', None)
        if mask is not None:
            m2 = F.adaptive_avg_pool2d(mask, c2.shape[-2:])
            m3 = F.adaptive_avg_pool2d(mask, c3.shape[-2:])
            m4 = F.adaptive_avg_pool2d(mask, c4.shape[-2:])
            conf = torch.cat([_flat(m2), _flat(m3), _flat(m4)], dim=1)  # (B,L,1)
        else:
            conf = torch.ones(B, c.shape[1], 1, device=c.device, dtype=c.dtype)
        return c, conf
```

- [ ] **Step 4: 執行確認通過**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_rpm.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/deform_adapter.py segment-anything/tests/test_rpm.py
git commit -m "feat(adapter): add ReferencePriorModule (UAWarpC SPM replacement, A3)"
```

---

### Task 4: Injector(可變形、detach query、softplus gate、信心加權 value)

**Files:**
- Modify: `segment-anything/segment_anything/modeling/deform_adapter.py`(append `Injector`)
- Test: `segment-anything/tests/test_deform_injector.py`

**Interfaces:**
- Consumes:`MSDeformAttn`、`deform_inputs` 的 inject 三元組、RPM 的 `(c, conf)`。
- Produces:`class Injector(dim=1280, n_heads=8, n_points=4, n_levels=3, deform_ratio=0.5, gate_init=_DEFAULT_GATE_INIT)`,
  `forward(x_tokens, c, conf, inject_inputs) -> x_tokens'`,`x_tokens`:(B, Lq=Hvit*Wvit, dim)。
  對 query `.detach()`;殘差保留梯度;`softplus(gate)` 縮放;value 進 attn 前乘 `conf`。

- [ ] **Step 1: 寫失敗測試**

```python
# segment-anything/tests/test_deform_injector.py
import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import Injector, ReferencePriorModule, deform_inputs


def _setup(dim=32, h=4):
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=dim)
    feats = {'l2': torch.randn(1, 8, h * 2, h * 2),
             'l3': torch.randn(1, 16, h, h),
             'mask': torch.rand(1, 1, h * 2, h * 2)}
    c, conf = rpm(feats)
    inj_in, _ = deform_inputs(h, h, torch.device('cpu'))
    x = torch.randn(1, h * h, dim)
    return Injector(dim=dim, n_heads=4, n_levels=3, deform_ratio=0.5), x, c, conf, inj_in


def test_inject_shape_preserved():
    inj, x, c, conf, inj_in = _setup()
    out = inj(x, c, conf, inj_in)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_gate_initial_value_approx_0_05():
    inj = Injector(dim=32, n_heads=4)
    assert abs(F.softplus(inj.gate).item() - 0.05) < 0.005


def test_query_detached_residual_only_grad():
    """Q detach → ∂sum(out)/∂x = 1（純殘差）。"""
    inj, _, c, conf, inj_in = _setup()
    x = torch.randn(1, 16, 32, requires_grad=True)
    inj(x, c, conf, inj_in).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad)), "Q 未 detach：grad 非全 1"


def test_low_confidence_weakens_injection():
    inj, x, c, conf, inj_in = _setup()
    with torch.no_grad():
        inj.gate.fill_(5.0)
    out_hi = inj(x.clone(), c, torch.ones_like(conf), inj_in)
    out_lo = inj(x.clone(), c, torch.zeros_like(conf), inj_in)
    # conf=0 → value 全 0 → 注入量應明顯小於 conf=1
    assert (out_lo - x).abs().mean() < (out_hi - x).abs().mean()
```

- [ ] **Step 2: 執行確認失敗**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_injector.py -v`
Expected: FAIL — `ImportError: cannot import name 'Injector'`

- [ ] **Step 3: 實作 Injector**

```python
class Injector(nn.Module):
    """ViT-Adapter Injector（可變形）。Q=ViT.detach()，K/V=多尺度 c（信心加權），softplus gate。"""
    def __init__(self, dim=1280, n_heads=8, n_points=4, n_levels=3,
                 deform_ratio=0.5, gate_init=_DEFAULT_GATE_INIT):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=n_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

    def forward(self, x_tokens, c, conf, inject_inputs):
        ref_pts, spatial_shapes, lsi = inject_inputs
        ref_pts = ref_pts.to(x_tokens.device)
        if ref_pts.shape[0] != x_tokens.shape[0]:
            ref_pts = ref_pts.expand(x_tokens.shape[0], -1, -1, -1)
        q = self.query_norm(x_tokens.detach())          # detach：不讓 adapter 梯度重塑 ViT
        feat = self.feat_norm(c) * conf                  # 信心加權 value（決策③）
        delta = self.attn(q, ref_pts, feat, spatial_shapes, lsi)
        gate = F.softplus(self.gate)
        return x_tokens + gate * delta                   # 殘差保留 ViT 梯度
```

- [ ] **Step 4: 執行確認通過**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_injector.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/deform_adapter.py segment-anything/tests/test_deform_injector.py
git commit -m "feat(adapter): add deformable Injector (detached query, confidence-weighted, A3)"
```

---

### Task 5: Extractor + ConvFFN(detach ViT feat)

**Files:**
- Modify: `segment-anything/segment_anything/modeling/deform_adapter.py`(append `DWConv`, `ConvFFN`, `Extractor`)
- Test: `segment-anything/tests/test_deform_extractor.py`

**Interfaces:**
- Consumes:`MSDeformAttn`、`deform_inputs` 的 extract 三元組、`c`(B,L,dim)、ViT tokens(B,Lvit,dim)、三尺度 grid 尺寸清單。
- Produces:
  - `class DWConv(dim)`,`forward(x, scale_hw: list[(H,W)]) -> x`(逐尺度深度卷積)。
  - `class ConvFFN(dim, hidden_ratio=0.25)`,`forward(x, scale_hw) -> x`。
  - `class Extractor(dim=1280, n_heads=8, n_points=4, deform_ratio=0.5, with_cffn=True, drop_path=0.)`,
    `forward(c, x_tokens, extract_inputs, scale_hw) -> c'`;對 ViT feat `.detach()`;`c` 保留梯度。

- [ ] **Step 1: 寫失敗測試**

```python
# segment-anything/tests/test_deform_extractor.py
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import (
    Extractor, ReferencePriorModule, deform_inputs)


def _setup(dim=32, h=4):
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=dim)
    c, _ = rpm({'l2': torch.randn(1, 8, h * 2, h * 2), 'l3': torch.randn(1, 16, h, h)})
    _, ext_in = deform_inputs(h, h, torch.device('cpu'))
    scale_hw = [(h * 2, h * 2), (h, h), (h // 2, h // 2)]
    x = torch.randn(1, h * h, dim)
    return Extractor(dim=dim, n_heads=4, deform_ratio=0.5), c, x, ext_in, scale_hw


def test_extract_updates_c_shape_preserved():
    ext, c, x, ext_in, scale_hw = _setup()
    c2 = ext(c, x, ext_in, scale_hw)
    assert c2.shape == c.shape and torch.isfinite(c2).all()


def test_vit_feat_detached_no_grad_to_vit():
    """K/V=ViT.detach() → ∂sum(c')/∂x 應為 None 或全 0。"""
    ext, c, _, ext_in, scale_hw = _setup()
    x = torch.randn(1, 16, 32, requires_grad=True)
    ext(c, x, ext_in, scale_hw).sum().backward()
    assert x.grad is None or x.grad.abs().max().item() == 0.0


def test_c_receives_gradient():
    ext, c, x, ext_in, scale_hw = _setup()
    c = c.clone().requires_grad_(True)
    ext(c, x, ext_in, scale_hw).sum().backward()
    assert c.grad is not None and c.grad.abs().max().item() > 0.0
```

- [ ] **Step 2: 執行確認失敗**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_extractor.py -v`
Expected: FAIL — `ImportError: cannot import name 'Extractor'`

- [ ] **Step 3: 實作 DWConv / ConvFFN / Extractor**

```python
class DWConv(nn.Module):
    """逐尺度深度卷積：把 concat 的 3 尺度 token 拆回各自 2D grid 各做 DWConv 再串回。"""
    def __init__(self, dim=1280):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, scale_hw):
        B, N, C = x.shape
        outs, start = [], 0
        for (H_, W_) in scale_hw:
            n = H_ * W_
            xi = x[:, start:start + n, :].transpose(1, 2).view(B, C, H_, W_)
            outs.append(self.dwconv(xi).flatten(2).transpose(1, 2))
            start += n
        return torch.cat(outs, dim=1)


class ConvFFN(nn.Module):
    def __init__(self, dim=1280, hidden_ratio=0.25):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, scale_hw):
        x = self.fc1(x)
        x = self.dwconv(x, scale_hw)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Extractor(nn.Module):
    """ViT-Adapter Extractor（可變形）。Q=c，K/V=ViT.detach()，+ ConvFFN 逐尺度精修 c。"""
    def __init__(self, dim=1280, n_heads=8, n_points=4, deform_ratio=0.5,
                 with_cffn=True, drop_path=0.):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=1, n_heads=n_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(dim=dim)
            self.ffn_norm = nn.LayerNorm(dim)
            self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, c, x_tokens, extract_inputs, scale_hw):
        ref_pts, spatial_shapes, lsi = extract_inputs
        ref_pts = ref_pts.to(c.device)
        if ref_pts.shape[0] != c.shape[0]:
            ref_pts = ref_pts.expand(c.shape[0], -1, -1, -1)
        feat = self.feat_norm(x_tokens.detach())         # ViT 當固定語境，不回灌梯度
        attn = self.attn(self.query_norm(c), ref_pts, feat, spatial_shapes, lsi)
        c = c + attn
        if self.with_cffn:
            c = c + self.drop_path(self.ffn(self.ffn_norm(c), scale_hw))
        return c
```

- [ ] **Step 4: 執行確認通過**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_extractor.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/deform_adapter.py segment-anything/tests/test_deform_extractor.py
git commit -m "feat(adapter): add deformable Extractor + ConvFFN (detached ViT feat, A3)"
```

---

### Task 6: DeformAdapter 協調器(狀態機 + hook 工廠)

**Files:**
- Modify: `segment-anything/segment_anything/modeling/deform_adapter.py`(append `DeformAdapter`)
- Test: `segment-anything/tests/test_deform_adapter.py`

**Interfaces:**
- Consumes:RPM / Injector / Extractor / deform_inputs。
- Produces:`class DeformAdapter(vit_dim=1280, l2_channels=256, l3_channels=512, n_heads=8, deform_ratio=0.5, use_reference=True)`,
  常數 `INJECT_BLOCKS=[0,8,16,24]`、`EXTRACT_BLOCKS=[7,15,23]`;
  `set_features(feats: dict, h: int, w: int)`;
  `_make_inject_pre_hook(stage_idx)` → pre-hook(回傳改動後 `(x,)`);
  `_make_extract_post_hook(stage_idx)` → post-hook(更新 `self._c`,回傳原 output);
  診斷欄位 `_last_gate_val` / `_last_inject_cos_sim`。

- [ ] **Step 1: 寫失敗測試**

```python
# segment-anything/tests/test_deform_adapter.py
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import DeformAdapter


def _adapter(dim=32, h=4):
    a = DeformAdapter(vit_dim=dim, l2_channels=8, l3_channels=16, n_heads=4)
    feats = {'l2': torch.randn(1, 8, h * 2, h * 2),
             'l3': torch.randn(1, 16, h, h),
             'mask': torch.rand(1, 1, h * 2, h * 2)}
    a.set_features(feats, h, h)
    return a, h, dim


def test_block_indices():
    a = DeformAdapter(vit_dim=32, l2_channels=8, l3_channels=16, n_heads=4)
    assert a.INJECT_BLOCKS == [0, 8, 16, 24]
    assert a.EXTRACT_BLOCKS == [7, 15, 23]
    assert len(a.injectors) == 4 and len(a.extractors) == 3


def test_inject_pre_hook_shape_preserved():
    a, h, dim = _adapter()
    hook = a._make_inject_pre_hook(0)
    x = torch.randn(1, h, h, dim)                      # SAM block I/O: (B,H,W,C)
    out = hook(None, (x,))
    assert isinstance(out, tuple) and out[0].shape == (1, h, h, dim)


def test_extract_post_hook_updates_c_returns_output_unchanged():
    a, h, dim = _adapter()
    c_before = a._c.clone()
    x = torch.randn(1, h, h, dim)
    hook = a._make_extract_post_hook(0)
    out = hook(None, None, x)
    assert torch.equal(out, x), "post-hook 必須回傳原 output 不變"
    assert not torch.equal(a._c, c_before), "c 必須被 extractor 更新"


def test_full_four_stage_sequence_runs():
    a, h, dim = _adapter()
    for s, blk in enumerate([0, 8, 16, 24]):
        x = torch.randn(1, h, h, dim)
        a._make_inject_pre_hook(s)(None, (x,))
        if s < 3:
            a._make_extract_post_hook(s)(None, None, torch.randn(1, h, h, dim))
    assert a._last_gate_val > 0.0
```

- [ ] **Step 2: 執行確認失敗**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter.py -v`
Expected: FAIL — `ImportError: cannot import name 'DeformAdapter'`

- [ ] **Step 3: 實作 DeformAdapter**

```python
class DeformAdapter(nn.Module):
    """雙向可變形 adapter 協調器：管理多尺度 c 狀態、4 injector + 3 extractor、hook 工廠。
    非侵入：透過 ViT block 的 forward pre/post hook 加殘差，不改 encoder 結構。"""
    INJECT_BLOCKS = [0, 8, 16, 24]
    EXTRACT_BLOCKS = [7, 15, 23]

    def __init__(self, vit_dim=1280, l2_channels=256, l3_channels=512,
                 n_heads=8, deform_ratio=0.5, use_reference=True):
        super().__init__()
        self.rpm = ReferencePriorModule(l2_channels, l3_channels, dim=vit_dim,
                                        use_reference=use_reference)
        self.injectors = nn.ModuleList([
            Injector(dim=vit_dim, n_heads=n_heads, n_levels=3, deform_ratio=deform_ratio)
            for _ in range(len(self.INJECT_BLOCKS))])
        self.extractors = nn.ModuleList([
            Extractor(dim=vit_dim, n_heads=n_heads, deform_ratio=deform_ratio)
            for _ in range(len(self.EXTRACT_BLOCKS))])
        self.use_reference = use_reference

        self._c = None
        self._conf = None
        self._inject_inputs = None
        self._extract_inputs = None
        self._scale_hw = None
        self._last_gate_val = float(F.softplus(torch.tensor(_DEFAULT_GATE_INIT)))
        self._last_inject_cos_sim = 1.0

    def set_features(self, feats, h, w):
        device = feats['l2'].device
        self._c, self._conf = self.rpm(feats)
        self._inject_inputs, self._extract_inputs = deform_inputs(h, w, device)
        self._scale_hw = [(h * 2, w * 2), (h, w), (h // 2, w // 2)]

    def _make_inject_pre_hook(self, stage_idx):
        def hook(module, inp):
            x = inp[0]
            B, H, W, C = x.shape
            tokens = x.reshape(B, H * W, C)
            out = self.injectors[stage_idx](tokens, self._c, self._conf, self._inject_inputs)
            with torch.no_grad():
                self._last_gate_val = float(F.softplus(self.injectors[stage_idx].gate).item())
                self._last_inject_cos_sim = float(
                    F.cosine_similarity(tokens, out, dim=-1).mean().item())
            return (out.reshape(B, H, W, C),)
        return hook

    def _make_extract_post_hook(self, stage_idx):
        def hook(module, inp, output):
            B, H, W, C = output.shape
            vit_tokens = output.reshape(B, H * W, C)
            self._c = self.extractors[stage_idx](
                self._c, vit_tokens, self._extract_inputs, self._scale_hw)
            return output  # 不改 ViT 輸出
        return hook
```

- [ ] **Step 4: 執行確認通過**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/deform_adapter.py segment-anything/tests/test_deform_adapter.py
git commit -m "feat(adapter): add DeformAdapter orchestrator with hook factories (A3)"
```

---

### Task 7: 接入 WeatherSAM 與 build 設定

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_sam.py`
  (import、`__init__` 建構、hook 註冊、`forward` 的 `set_features` 呼叫)
- Modify: `segment-anything/segment_anything/build_weather_sam.py`(建構新 adapter)
- Test: `segment-anything/tests/test_deform_adapter_integration.py`

**Interfaces:**
- Consumes:`DeformAdapter`;`fusion_module.pre_align(img_curr, img_ref)` 回傳 `{'l2','l3','mask'}`。
- Produces:`WeatherSAM.enable_deform_adapter()` / `disable_deform_adapter()`;
  `forward` 在 encoder 前呼叫 `self.vgg_injector.set_features(feats, h, w)`(h=w=img_size//16)。

實作說明(取代既有 `MultiScaleCrossAttnInjector` 相關接線,行號以現況為準):

`weather_sam.py` import 區(第 ~10 行附近):把
`from .vgg_adapter import MultiScaleCrossAttnInjector` 改為
`from .deform_adapter import DeformAdapter`。

`__init__`(現況第 84–85 行 `self.vgg_injector = MultiScaleCrossAttnInjector(...)`)改為:

```python
        _vit_dim = image_encoder.patch_embed.proj.out_channels
        self.vgg_injector = DeformAdapter(
            vit_dim=_vit_dim, l2_channels=256, l3_channels=512,
            n_heads=8, deform_ratio=0.5, use_reference=True,
        )
```

hook 註冊(現況第 135–157 `enable_vgg_adapter`)改為配對註冊:

```python
    def enable_deform_adapter(self):
        for handle in self._adapter_hook_handles:
            handle.remove()
        self._adapter_hook_handles = []
        n_blocks = len(self.image_encoder.blocks)
        for s, blk_idx in enumerate(self.vgg_injector.INJECT_BLOCKS):
            if blk_idx >= n_blocks:
                continue
            h = self.image_encoder.blocks[blk_idx].register_forward_pre_hook(
                self.vgg_injector._make_inject_pre_hook(s))
            self._adapter_hook_handles.append(h)
        for s, blk_idx in enumerate(self.vgg_injector.EXTRACT_BLOCKS):
            if blk_idx >= n_blocks:
                continue
            h = self.image_encoder.blocks[blk_idx].register_forward_hook(
                self.vgg_injector._make_extract_post_hook(s))
            self._adapter_hook_handles.append(h)
        self.use_vgg_adapter = True
        print(f'[WeatherSAM] Deform Adapter enabled: inject@{self.vgg_injector.INJECT_BLOCKS}, '
              f'extract@{self.vgg_injector.EXTRACT_BLOCKS}.')

    def disable_deform_adapter(self):
        for handle in self._adapter_hook_handles:
            handle.remove()
        self._adapter_hook_handles = []
        self.use_vgg_adapter = False
```

`forward` 設定特徵(現況第 205–206):把
`self.vgg_injector.set_features(_vgg_ref_aligned)` 改為帶 h,w:

```python
            if self.use_vgg_adapter and _vgg_ref_aligned is not None:
                _grid = self.image_encoder.img_size // self.image_encoder.patch_embed.proj.stride[0]
                self.vgg_injector.set_features(_vgg_ref_aligned, _grid, _grid)
```

`build_weather_sam.py`:找到呼叫 `enable_vgg_adapter(...)` / 舊 adapter enable 的位置,改呼叫
`model.enable_deform_adapter()`(其餘 `use_vgg_adapter` gate 邏輯不變)。

- [ ] **Step 1: 寫失敗整合測試(mock pre_align，避免載入 CMA 權重)**

```python
# segment-anything/tests/test_deform_adapter_integration.py
"""整合：DeformAdapter 掛上真實 SAM ViT-H block，全 forward 輸出形狀不變、無 NaN、ViT 受保護。
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter_integration.py -v
"""
import torch, torch.nn as nn, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.deform_adapter import DeformAdapter
from functools import partial


def _tiny_encoder(dim=32, depth=32, img=64, patch=16):
    return ImageEncoderViT(
        img_size=img, patch_size=patch, embed_dim=dim, depth=depth, num_heads=4,
        out_chans=16, global_attn_indexes=(7, 15, 23, 31),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=2)


def test_forward_output_shape_unchanged_with_adapter():
    enc = _tiny_encoder()
    grid = 64 // 16
    ad = DeformAdapter(vit_dim=32, l2_channels=8, l3_channels=16, n_heads=4)
    handles = []
    for s, b in enumerate(ad.INJECT_BLOCKS):
        handles.append(enc.blocks[b].register_forward_pre_hook(ad._make_inject_pre_hook(s)))
    for s, b in enumerate(ad.EXTRACT_BLOCKS):
        handles.append(enc.blocks[b].register_forward_hook(ad._make_extract_post_hook(s)))

    x = torch.randn(1, 3, 64, 64)
    out_noadapter = _tiny_encoder_forward_ref(enc, x)  # 形狀基準
    ad.set_features({'l2': torch.randn(1, 8, grid * 2, grid * 2),
                     'l3': torch.randn(1, 16, grid, grid),
                     'mask': torch.rand(1, 1, grid * 2, grid * 2)}, grid, grid)
    out = enc(x)
    for h in handles:
        h.remove()
    assert out.shape == out_noadapter.shape
    assert torch.isfinite(out).all()


def _tiny_encoder_forward_ref(enc, x):
    with torch.no_grad():
        return enc(x)
```

- [ ] **Step 2: 執行確認失敗**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter_integration.py -v`
Expected: FAIL(先因尚未套用 weather_sam 接線/或 shape 對不上而失敗)

- [ ] **Step 3: 套用 weather_sam.py 與 build_weather_sam.py 修改(如上 Interfaces 說明),並使測試通過**

依上述 import / `__init__` / `enable_deform_adapter` / `forward` / build 五處修改。整合測試本身
直接掛 hook 驗證,不依賴 build;完成後測試應綠。

- [ ] **Step 4: 執行確認通過**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter_integration.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: Commit**

```bash
git add segment-anything/segment_anything/modeling/weather_sam.py segment-anything/segment_anything/build_weather_sam.py segment-anything/tests/test_deform_adapter_integration.py
git commit -m "feat(adapter): wire DeformAdapter into WeatherSAM (inject/extract hooks, A3)"
```

---

### Task 8: 端到端 smoke + 記憶體 dry-run(4090)

**Files:**
- Create: `segment-anything/tests/test_deform_adapter_e2e.py`
- Create: `segment-anything/scripts/deform_adapter_memcheck.py`

**Interfaces:**
- Consumes:`build_weather_sam_from_config`(既有)。
- Produces:e2e 測試(輸出 key 與舊版一致)+ 記憶體量測腳本(4090 上人工執行)。

- [ ] **Step 1: 寫 e2e 測試(CPU、tiny config、確認 forward 產出 masks/low_res_logits/class_ids)**

```python
# segment-anything/tests/test_deform_adapter_e2e.py
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.build_weather_sam import build_weather_sam_from_config


def test_full_model_forward_keys_unchanged():
    cfg = {"model_type": "vit_h", "use_vgg_adapter": True, "cond": True,
           "lrh": True, "decoder": "unified", "ref": True, "mfb": True}
    model = build_weather_sam_from_config(cfg, checkpoint=None).eval()
    B, S = 1, model.image_encoder.img_size
    batch = [{
        "image": torch.randint(0, 255, (3, S, S)).float(),
        "clear_image": torch.randint(0, 255, (3, S, S)).float(),
        "text_prompts": ["road", "car"],
        "condition_id": torch.tensor(0),
        "original_size": (S, S),
    }]
    with torch.no_grad():
        out = model(batch)
    assert set(out[0].keys()) == {"masks", "low_res_logits", "class_ids"}
    assert torch.isfinite(out[0]["low_res_logits"]).all()
```

- [ ] **Step 2: 執行確認(先失敗再修至通過)**

Run: `conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter_e2e.py -v`
Expected: 最終 PASS(1 passed)。若 build 端仍呼叫舊 enable,修正為 `enable_deform_adapter()`。

- [ ] **Step 3: 寫 4090 記憶體量測腳本**

```python
# segment-anything/scripts/deform_adapter_memcheck.py
"""4090 記憶體 dry-run：1024²、bf16、grad ckpt、3 步 backward，印峰值記憶體。
執行：conda run -n sam_env python segment-anything/scripts/deform_adapter_memcheck.py
若 OOM，依 spec §7 緩解階梯：with_cp → deform_ratio → 1/8 預 pool → 退 2 尺度。"""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.build_weather_sam import build_weather_sam_from_config

cfg = {"model_type": "vit_h", "use_vgg_adapter": True, "cond": True,
       "lrh": True, "decoder": "unified", "ref": True, "mfb": True}
model = build_weather_sam_from_config(cfg, checkpoint="checkpoints/sam_vit_h_4b8939.pth").cuda()
model.image_encoder.use_checkpoint = True
S = model.image_encoder.img_size
batch = [{
    "image": torch.randint(0, 255, (3, S, S)).float(),
    "clear_image": torch.randint(0, 255, (3, S, S)).float(),
    "text_prompts": ["road", "car", "person", "building"],
    "condition_id": torch.tensor(0),
    "original_size": (S, S),
}]
torch.cuda.reset_peak_memory_stats()
for step in range(3):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
        loss = out[0]["low_res_logits"].float().mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    print(f"[step {step}] peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("OK: no OOM")
```

- [ ] **Step 4: 4090 上人工執行記憶體腳本**

Run: `conda run -n sam_env python segment-anything/scripts/deform_adapter_memcheck.py`
Expected: 印出三步峰值記憶體且 "OK: no OOM"。若 OOM,依 spec §7 緩解階梯調整並記錄於本檔。

- [ ] **Step 5: Commit**

```bash
git add segment-anything/tests/test_deform_adapter_e2e.py segment-anything/scripts/deform_adapter_memcheck.py
git commit -m "test(adapter): add e2e forward test + 4090 memory dry-run (A3)"
```

---

## Follow-up(非本計畫範圍,另開)
- `SameImageAdapterInjector` 基線同步加 extractor 鏡像,以與新 FULL 對照。
- 舊 `vgg_adapter.py::MultiScaleCrossAttnInjector` 是否保留為 legacy ablation:先保留不刪,待新版驗證後另議。
- MSDeformAttn CUDA 加速版編譯(可選優化,非必要)。

## Self-Review 檢核(對照 spec)
- spec §3 架構(4 inj + 3 ext、pre@0/8/16/24 + post@7/15/23):Task 2/6/7 覆蓋。
- spec §4.1 RPM(3 尺度、1/32 由 l3 降採、level_embed、use_reference 零化):Task 3 覆蓋。
- spec §4.2 Injector(detach query、softplus gate、信心乘 value、deform_ratio=0.5):Task 4 覆蓋。
- spec §4.3 Extractor(detach ViT feat、ConvFFN 逐尺度 DWConv、末組不 extract):Task 5/6 覆蓋。
- spec §4.4 梯度保護:Task 4(query detach)、Task 5(feat detach)測試覆蓋。
- spec §5 hook 生命週期(set_features、post-hook 回傳原 output):Task 6 覆蓋。
- spec §7 記憶體 dry-run:Task 8 覆蓋。
- spec §9 ablation(use_vgg_adapter=False 不掛 hook、use_reference 零化):Task 3(零化)、Task 7(gate 不掛)覆蓋。
- 型別一致:`set_features(feats, h, w)`、`_make_inject_pre_hook`/`_make_extract_post_hook`、
  `deform_inputs` 三元組介面跨 Task 一致。
