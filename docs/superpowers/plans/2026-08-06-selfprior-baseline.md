# 同影像先驗基線（`--prior_source self`）實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `--prior_source self` 旗標，使 DeformAdapter 的先驗取自當前影像而非 UAWarpC 對齊後的晴天參考，並跑出可放入論文 ACDC 比較表的受控基線數據。

**Architecture:** 於 `CMAAlignment` 新增 `self_prior()`，鏡像 `pre_align()` 的輸出契約但跳過 UAWarpC，且不回傳 `'mask'` 鍵。`ReferencePriorModule` 既有的 `feats.get('mask', None)` 分支會因此自動取得 `conf ≡ 1`，故 `deform_adapter.py` 完全不需修改。`PairSAM` 將 forward 階段 0 抽成 `_build_adapter_prior()` 方法後依 `prior_source` 分派，builder 與 CLI 逐層透傳。

**Tech Stack:** PyTorch、conda 環境 `sam_env`、pytest。單卡 RTX 4090 24GB。

**設計文件：** `docs/superpowers/specs/2026-08-06-selfprior-baseline-design.md`

## Global Constraints

- 所有 Python 指令一律在 conda 環境 `sam_env` 執行：`conda run -n sam_env python ...`
- 工作目錄為 repo 根目錄 `/home/rvl1421/SAM_research-1`；`train.py` 與 `scripts/` 位於 `segment-anything/` 之下
- **`--prior_source` 預設值為 `reference`，既有行為必須零變動。** 每個 Task 都要有回歸測試證明這點
- **`segment_anything/modeling/deform_adapter.py` 不得修改。** `conf ≡ 1` 必須靠省略 `'mask'` 鍵走既有 else 分支達成
- **不得設定 `_adapter_reference_free = True`。** 該旗標供 `adapter_variant='sam_adapter'`（W4）使用，會整個替換 `vgg_injector`
- **不得修改 `paper/` 之下任何檔案。** 本計畫只產出實驗數據
- 訓練協定完全沿用 `scripts/ablation_m2f_common.sh` 的 `BASE_FLAGS`，一個旗標都不覆寫
- Run ID 固定為 `P1_selfprior_seed42`
- **數值一律以 `e1_results.json` 的 `overall_miou` 為準**，不得以 `train_log.csv` 的 val 峰值與既有 run 相比（FULL 兩者為 76.02 對 76.10）
- 對照基準（取自各 run 的 `e1_results.json`）：`FULL_seed42` = 76.02、`W2_semB_seed42` = 76.50、`W4_seed42` = 79.80

---

## File Structure

| 檔案 | 職責 | 動作 |
|---|---|---|
| `segment-anything/segment_anything/modeling/fusion.py` | `CMAAlignment.self_prior()`：當前影像 VGG 多尺度先驗 | 修改 |
| `segment-anything/segment_anything/modeling/pair_sam.py` | `prior_source` 屬性 + `_build_adapter_prior()` 分派 | 修改 |
| `segment-anything/segment_anything/build_pair_sam.py` | cfg → `model.prior_source` 映射 | 修改 |
| `segment-anything/train.py` | `--prior_source` CLI 旗標，寫入 `abl_cfg` | 修改 |
| `segment-anything/scripts/memcheck_m2f.py` | 加 `--prior_source` 供 smoke 使用 | 修改 |
| `segment-anything/scripts/ablation_m2f_phase6_selfprior.sh` | Phase 6 執行腳本 | 建立 |
| `segment-anything/tests/test_self_prior.py` | Task 1 測試 | 建立 |
| `segment-anything/tests/test_prior_source_switch.py` | Task 2、3 測試 | 建立 |
| `docs/experiments/2026-08-06-selfprior-baseline.md` | 結果報告 | 建立 |

`segment_anything/modeling/deform_adapter.py` 刻意不列入 —— 它必須保持不變。

---

### Task 1: `CMAAlignment.self_prior()`

**Files:**
- Modify: `segment-anything/segment_anything/modeling/fusion.py`（在 `pre_align()` 之後新增方法，`pre_align` 定義於 166–285 行）
- Test: `segment-anything/tests/test_self_prior.py`（建立）

**Interfaces:**
- Consumes: `CMAAlignment._extract_vgg_features(img) -> (feats, feats_256)`，其中 `feats` 為 5 元素 list，index 2 = 256ch stride-8、index 3 = 512ch stride-16、index 4 = 512ch stride-32
- Produces: `CMAAlignment.self_prior(img_curr, out_size=(64, 64), l2_native=False) -> dict`，鍵為 `'l2'`（256ch）、`'l3'`（512ch）、`'l4'`（512ch），**無 `'mask'` 鍵**。`l2_native=True` 時 `l2` 為 `(2*out_H, 2*out_W)`、`l3` 為 `(out_H, out_W)`、`l4` 為 `(out_H//2, out_W//2)`

- [ ] **Step 1: 建立測試檔並寫入失敗測試**

建立 `segment-anything/tests/test_self_prior.py`：

```python
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_self_prior.py -v

self_prior() 的語義鎖定：先驗取自當前影像、不經 UAWarpC、無 'mask' 鍵
（使 ReferencePriorModule 走既有 no-mask 分支取得 conf≡1）。

以 256×256 輸入 + out_size=(16,16) 讓 VGG16 maxpool 管線成立且 CPU 執行時間短，
與 tests/test_pre_align_native_l2.py 同一慣例。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from segment_anything.modeling.fusion import CMAAlignment
from segment_anything.modeling.deform_adapter import ReferencePriorModule


def make_model():
    """隨機初始化的 CMAAlignment —— 只驗形狀與語義，不需預訓練權重。"""
    return CMAAlignment(embed_dim=256, pretrained_path=None)


def make_image(H=256, W=256, B=1):
    return torch.rand(B, 3, H, W) * 255.0


def test_self_prior_shapes_match_pre_align():
    """三個尺度的 shape 與 channel 必須逐一等同 pre_align(l2_native=True)。"""
    model = make_model()
    img_curr, img_ref = make_image(), make_image()
    ref_out = model.pre_align(img_curr, img_ref, out_size=(16, 16), l2_native=True)
    self_out = model.self_prior(img_curr, out_size=(16, 16), l2_native=True)
    for k in ('l2', 'l3', 'l4'):
        assert self_out[k].shape == ref_out[k].shape, (
            f"{k} shape 不對等：self={self_out[k].shape} vs ref={ref_out[k].shape}")


def test_self_prior_omits_mask_key():
    """不得回傳 'mask'；RPM 靠 feats.get('mask', None) is None 取得 conf≡1。"""
    model = make_model()
    self_out = model.self_prior(make_image(), out_size=(16, 16), l2_native=True)
    assert 'mask' not in self_out


def test_self_prior_uses_current_image_features():
    """l3 必須等於當前影像 VGG index-3 特徵縮放後的結果，不含任何翹曲。"""
    model = make_model()
    img = make_image()
    with torch.no_grad():
        feats, _ = model._extract_vgg_features(img)
        expected = F.interpolate(feats[3], size=(16, 16),
                                 mode='bilinear', align_corners=False)
    out = model.self_prior(img, out_size=(16, 16), l2_native=True)
    assert torch.allclose(out['l3'], expected, atol=1e-5)


def test_self_prior_is_independent_of_reference_image():
    """同一張當前影像 → 輸出恆定，與是否存在參考影像無關。"""
    model = make_model()
    img = make_image()
    a = model.self_prior(img, out_size=(16, 16), l2_native=True)
    b = model.self_prior(img, out_size=(16, 16), l2_native=True)
    assert torch.allclose(a['l2'], b['l2'], atol=1e-6)


def test_self_prior_sets_neutral_telemetry():
    """pair_trainer.py 讀 _last_conf_mean/_last_valid_ratio；self 模式須為中性值 1.0，
    否則 train_log.csv 欄位語義與其他 run 不一致。"""
    model = make_model()
    model._last_conf_mean = 0.3
    model._last_valid_ratio = 0.3
    model._last_flow = torch.zeros(1, 2, 16, 16)
    model._last_confidence_map = torch.zeros(1, 1, 16, 16)
    model.self_prior(make_image(), out_size=(16, 16), l2_native=True)
    assert model._last_conf_mean == 1.0
    assert model._last_valid_ratio == 1.0
    assert model._last_flow is None
    assert model._last_confidence_map is None


def test_rpm_returns_neutral_conf_without_mask_key():
    """契約鎖定：feats 無 'mask' 鍵 → conf≡1 且參考特徵 c 保持非零。
    這是 deform_adapter.py 不需修改的原因。"""
    torch.manual_seed(0)
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=8, l4_channels=8, dim=16)
    feats = {
        'l2': torch.randn(1, 8, 8, 8),
        'l3': torch.randn(1, 8, 4, 4),
        'l4': torch.randn(1, 8, 2, 2),
    }
    c, conf = rpm(feats)
    assert torch.equal(conf, torch.ones_like(conf)), "無 mask 鍵時 conf 必須≡1"
    assert c.abs().sum() > 0, "先驗特徵不得被歸零（那是 --no-ref 的語義）"


def _small_rpm_and_feats():
    torch.manual_seed(0)
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=8, l4_channels=8, dim=16)
    feats = {
        'l2': torch.randn(1, 8, 8, 8),
        'l3': torch.randn(1, 8, 4, 4),
        'l4': torch.randn(1, 8, 2, 2),
    }
    return rpm, feats


def test_rpm_projections_receive_gradient_without_mask():
    """梯度連通：self 模式下 proj_c2/c3/c4 必須收到非零梯度。

    這是 self 與 --no-ref 的關鍵區別。--no-ref 走 c = zeros_like(c)，
    proj 卷積的輸出被丟棄、梯度斷聯（Phase 5 腳本記載的良性 Grad Audit 警報）；
    self 模式的先驗特徵實際參與注入，斷聯即代表接線錯誤。
    """
    rpm, feats = _small_rpm_and_feats()
    c, conf = rpm(feats)
    (c * conf).sum().backward()
    for name in ('proj_c2', 'proj_c3', 'proj_c4'):
        grad = getattr(rpm, name).weight.grad
        assert grad is not None, f"{name} 未收到梯度"
        assert grad.abs().sum() > 0, f"{name} 梯度全為零"


def test_no_ref_projections_stay_disconnected():
    """對照鎖定：use_reference=False 時 proj 卷積梯度斷聯，確認上一個測試
    量到的是真實差異而非恆真斷言。"""
    rpm, feats = _small_rpm_and_feats()
    rpm.use_reference = False
    c, conf = rpm(feats)
    (c * conf).sum().backward()
    grad = rpm.proj_c2.weight.grad
    assert grad is None or grad.abs().sum() == 0
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/test_self_prior.py -v
```

Expected: 五個 `self_prior` 測試 FAIL，訊息為 `AttributeError: 'CMAAlignment' object has no attribute 'self_prior'`。三個 `rpm` 測試（`test_rpm_returns_neutral_conf_without_mask_key`、`test_rpm_projections_receive_gradient_without_mask`、`test_no_ref_projections_stay_disconnected`）應 PASS —— 它們鎖定的是既有契約，正是 `deform_adapter.py` 不需修改的依據。

- [ ] **Step 3: 實作 `self_prior()`**

在 `segment-anything/segment_anything/modeling/fusion.py` 的 `pre_align()` 方法之後（第 285 行 `return {...}` 結束處之後、`FlowGuidedSemanticAlignment` 區塊註解之前）加入：

```python
    @torch.no_grad()
    def self_prior(
        self,
        img_curr: torch.Tensor,          # (B, 3, H, W) 當前影像，值域 [0, 255]
        out_size: tuple = (64, 64),      # 目標特徵圖空間尺寸
        l2_native: bool = False,         # True 時 l2 回傳 2×out_size（真 stride-8）
    ) -> dict:
        """同影像先驗（ViT-Adapter 式 SPM）：以當前影像的 VGG 多尺度特徵作為
        DeformAdapter 的先驗，不執行 UAWarpC 對齊。

        與 pre_align() 的輸出契約逐一對等（相同鍵、相同 channel、相同空間尺寸），
        唯一差異是**不回傳 'mask' 鍵**：無翹曲即無對齊不確定性，
        ReferencePriorModule 會因 feats.get('mask', None) is None 走既有 else 分支
        取得 conf ≡ 1。此為 ViT-Adapter 設定的正確實例化，非移除置信度調變。

        Returns:
            dict，鍵為 'l2'(256ch) / 'l3'(512ch) / 'l4'(512ch)，無 'mask'。
        """
        out_H, out_W = out_size
        feats_curr, _ = self._extract_vgg_features(img_curr)

        def _resize(x, size):
            if x.shape[-2:] != size:
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            return x

        # l2：l2_native=True 時為 2×out_size（真 stride-8），與 pre_align Step 9 對齊
        l2_size = (out_H * 2, out_W * 2) if l2_native else (out_H, out_W)
        f_l2 = _resize(feats_curr[2], l2_size)              # 256ch
        f_l3 = _resize(feats_curr[3], (out_H, out_W))       # 512ch
        f_l4 = _resize(feats_curr[4], (out_H // 2, out_W // 2))  # 512ch，真 stride-32

        # 遙測：pair_trainer.py 讀取這四個屬性。self 模式無對齊，置信度與有效率
        # 皆為中性值 1.0；flow 與 confidence map 無意義，設為 None。
        self._last_conf_mean = 1.0
        self._last_valid_ratio = 1.0
        self._last_flow = None
        self._last_confidence_map = None

        return {'l2': f_l2, 'l3': f_l3, 'l4': f_l4}
```

- [ ] **Step 4: 執行測試確認通過**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/test_self_prior.py -v
```

Expected: 8 passed。

- [ ] **Step 5: 回歸測試 —— 確認 `pre_align` 未受影響**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/test_pre_align_native_l2.py \
    segment-anything/tests/test_rpm.py \
    segment-anything/tests/test_ablation_flags_w3_w6.py -v
```

Expected: 全部 passed。

- [ ] **Step 6: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/segment_anything/modeling/fusion.py segment-anything/tests/test_self_prior.py
git commit -m "feat(fusion): add CMAAlignment.self_prior for same-image adapter prior

Mirrors pre_align's output contract (same keys, channels and spatial sizes)
but skips UAWarpC alignment and omits the 'mask' key. ReferencePriorModule
already falls back to conf=1 when 'mask' is absent, so deform_adapter.py
needs no change.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `PairSAM.prior_source` 分派與 builder 透傳

**Files:**
- Modify: `segment-anything/segment_anything/modeling/pair_sam.py`（`__init__` 約 41–56 行加屬性；forward 階段 0 於 204–226 行抽成方法）
- Modify: `segment-anything/segment_anything/build_pair_sam.py`（`build_pair_sam_from_config` 的 rpm 區塊，約 60–76 行）
- Test: `segment-anything/tests/test_prior_source_switch.py`（建立）

**Interfaces:**
- Consumes: Task 1 的 `CMAAlignment.self_prior(img_curr, out_size, l2_native) -> dict`
- Produces:
  - `PairSAM.prior_source: str`，預設 `'reference'`，合法值 `'reference'` / `'self'`
  - `PairSAM._build_adapter_prior(batched_input: List[dict]) -> Optional[dict]`，回傳給 `vgg_injector.set_features()` 的先驗 dict，不適用時回傳 `None`
  - `build_pair_sam_from_config(cfg, ...)` 讀取 `cfg['prior_source']`（預設 `'reference'`）並設到 `model.prior_source`

- [ ] **Step 1: 寫入失敗測試**

建立 `segment-anything/tests/test_prior_source_switch.py`：

```python
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_prior_source_switch.py -v

--prior_source 的語義鎖定：
  reference（預設）= UAWarpC 對齊後的跨視角參考先驗，既有行為
  self            = 當前影像的 VGG 多尺度先驗（ViT-Adapter 式 SPM），conf≡1

_build_adapter_prior 以 stub 直測，避免建構完整 ViT-H 前向（CPU 上過慢）。
builder 測試沿用 tests/test_build_from_config.py 的 vit_b + checkpoint=None 慣例（約 4s）。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

import torch

from segment_anything.modeling.pair_sam import PairSAM
from segment_anything.build_pair_sam import build_pair_sam_from_config


class _RecordingFusion:
    """記錄呼叫了哪個方法，回傳形狀無關緊要的假先驗。"""

    def __init__(self):
        self.calls = []

    def pre_align(self, img_curr, img_ref, out_size, l2_native=False):
        self.calls.append('pre_align')
        return {'l2': torch.zeros(1), 'l3': torch.zeros(1),
                'l4': torch.zeros(1), 'mask': torch.zeros(1)}

    def self_prior(self, img_curr, out_size, l2_native=False):
        self.calls.append('self_prior')
        return {'l2': torch.zeros(1), 'l3': torch.zeros(1), 'l4': torch.zeros(1)}


def _stub(prior_source='reference'):
    """以未實例化 PairSAM 的方式直測 _build_adapter_prior 的分派邏輯。"""
    obj = SimpleNamespace(
        use_vgg_adapter=True,
        _adapter_reference_free=False,
        prior_source=prior_source,
        fusion_module=_RecordingFusion(),
        device=torch.device('cpu'),
        image_encoder=SimpleNamespace(
            img_size=1024,
            patch_embed=SimpleNamespace(proj=SimpleNamespace(stride=(16, 16))),
        ),
    )
    obj._build_adapter_prior = PairSAM._build_adapter_prior.__get__(obj)
    return obj


def _batched(with_clear=True):
    rec = {'image': torch.zeros(3, 1024, 1024)}
    if with_clear:
        rec['clear_image'] = torch.zeros(3, 1024, 1024)
    return [rec]


def test_default_prior_source_dispatches_to_pre_align():
    obj = _stub('reference')
    out = obj._build_adapter_prior(_batched())
    assert obj.fusion_module.calls == ['pre_align']
    assert 'mask' in out


def test_self_prior_source_dispatches_to_self_prior():
    obj = _stub('self')
    out = obj._build_adapter_prior(_batched())
    assert obj.fusion_module.calls == ['self_prior']
    assert 'mask' not in out


def test_self_mode_does_not_require_clear_image():
    """self 模式不需要參考影像；缺 clear_image 仍須建出先驗。"""
    obj = _stub('self')
    out = obj._build_adapter_prior(_batched(with_clear=False))
    assert obj.fusion_module.calls == ['self_prior']
    assert out is not None


def test_reference_mode_without_clear_image_returns_none():
    """reference 模式缺 clear_image 時維持既有行為：不注入先驗。"""
    obj = _stub('reference')
    out = obj._build_adapter_prior(_batched(with_clear=False))
    assert out is None
    assert obj.fusion_module.calls == []


def test_precomputed_embedding_skips_prior():
    """已有 image_embedding 時不重建先驗（既有行為）。"""
    obj = _stub('self')
    out = obj._build_adapter_prior([{'image_embedding': torch.zeros(1),
                                     'image': torch.zeros(3, 1024, 1024)}])
    assert out is None
    assert obj.fusion_module.calls == []


def _cfg(**over):
    base = dict(model_type='vit_b', use_vgg_adapter=True, inject='pre',
                decoder='unified', lrh=True, mfb=True, ref=True)
    base.update(over)
    return base


def test_builder_default_is_reference():
    m = build_pair_sam_from_config(_cfg(), checkpoint=None)
    assert m.prior_source == 'reference'


def test_builder_maps_prior_source_self():
    """關鍵：eval 與 test dump 都經 load_pair_sam_from_ablation → build_pair_sam_from_config
    重建模型。若此映射缺失，評估會靜默地以 reference 模式跑，得到錯誤數字。"""
    m = build_pair_sam_from_config(_cfg(prior_source='self'), checkpoint=None)
    assert m.prior_source == 'self'


def test_builder_rejects_unknown_prior_source():
    try:
        build_pair_sam_from_config(_cfg(prior_source='bogus'), checkpoint=None)
    except ValueError as e:
        assert 'prior_source' in str(e)
    else:
        raise AssertionError("未知 prior_source 必須拋 ValueError")


def test_self_mode_does_not_set_adapter_reference_free():
    """_adapter_reference_free 是 sam_adapter(W4) 專用，會整個替換 vgg_injector。
    self 模式必須保留 RPM 路徑。"""
    m = build_pair_sam_from_config(_cfg(prior_source='self'), checkpoint=None)
    assert m._adapter_reference_free is False
    assert hasattr(m.vgg_injector, 'rpm')


def test_self_mode_keeps_zero_init_gate():
    """閘控必須與 FULL 相同的零初始化，否則 P1 與 FULL 之間多一個變因。"""
    m = build_pair_sam_from_config(_cfg(prior_source='self'), checkpoint=None)
    for inj in m.vgg_injector.injectors:
        assert torch.equal(inj.gamma.detach(), torch.zeros_like(inj.gamma))
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/test_prior_source_switch.py -v
```

Expected: FAIL，`AttributeError: type object 'PairSAM' has no attribute '_build_adapter_prior'`。

- [ ] **Step 3: 於 `pair_sam.py` 加屬性與分派方法**

在 `segment-anything/segment_anything/modeling/pair_sam.py` 的 `__init__` 中，緊接 `self._adapter_reference_free: bool = False`（第 110 行）之後加入：

```python
        # 先驗來源：'reference' = UAWarpC 對齊後的跨視角晴天參考（FULL，預設）；
        # 'self' = 當前影像的 VGG 多尺度特徵（ViT-Adapter 式 SPM 基線）。
        # 由 build_pair_sam_from_config 依 cfg['prior_source'] 覆蓋。
        self.prior_source: str = 'reference'
```

在 `forward()` 之前新增方法：

```python
    def _build_adapter_prior(self, batched_input):
        """建構 DeformAdapter 的多尺度先驗，依 self.prior_source 分派。

        'reference'：pre_align(當前影像, 晴天參考) → 含 'mask'（對齊置信度）
        'self'     ：self_prior(當前影像)          → 無 'mask'（RPM 取 conf≡1）

        不適用時回傳 None（未啟用 adapter、reference-free 變體、已有預計算
        embedding、缺必要輸入）。
        """
        if not (self.use_vgg_adapter and not self._adapter_reference_free):
            return None
        if "image_embedding" in batched_input[0]:
            return None
        if not all("image" in x for x in batched_input):
            return None
        if self.prior_source == 'reference' and not all(
                "clear_image" in x for x in batched_input):
            return None

        img_curr_batch = torch.stack(
            [x["image"] for x in batched_input], dim=0).to(self.device)
        _grid = self.image_encoder.img_size // self.image_encoder.patch_embed.proj.stride[0]

        if self.prior_source == 'self':
            feats = self.fusion_module.self_prior(
                img_curr_batch, out_size=(_grid, _grid), l2_native=True)
        else:
            img_ref_batch = torch.stack(
                [x["clear_image"] for x in batched_input], dim=0).to(self.device)
            feats = self.fusion_module.pre_align(
                img_curr_batch, img_ref_batch, out_size=(_grid, _grid), l2_native=True)

        # pre_align/self_prior 在 no_grad 下產生大量 VGG 中間特徵（1024×1024）。
        # 釋放 CUDA allocator cache，為後續 ViT-H global attention 騰出空間。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return feats
```

將 forward 中原本的階段 0 區塊（204–226 行，自 `_vgg_ref_aligned = None` 起至 `torch.cuda.empty_cache()` 止）整段替換為：

```python
        # --- 階段 0：建構 DeformAdapter 先驗（依 prior_source 分派）---
        # CMAAlignment 僅使用原始影像的 VGG 特徵，不依賴 ViT embedding，
        # 因此可在 SAM Encoder 之前安全執行，不存在循環依賴。
        _vgg_ref_aligned = self._build_adapter_prior(batched_input)
```

階段 1 中 `if self.use_vgg_adapter and _vgg_ref_aligned is not None:` 一行維持不變，但需注意 `_grid` 原本在階段 0 計算、階段 1 使用。改為在階段 1 就地重算：

```python
            if self.use_vgg_adapter and _vgg_ref_aligned is not None:
                _grid = (self.image_encoder.img_size
                         // self.image_encoder.patch_embed.proj.stride[0])
                self.vgg_injector.set_features(_vgg_ref_aligned, _grid, _grid)
```

- [ ] **Step 4: 於 `build_pair_sam.py` 加 cfg 映射**

在 `segment-anything/segment_anything/build_pair_sam.py` 的 `build_pair_sam_from_config` 中，於 `_use_ref = bool(cfg.get('ref', True))`（第 60 行）之前加入：

```python
    # 先驗來源（P1 基線）：'self' = 同影像先驗（ViT-Adapter 式 SPM），不經 UAWarpC。
    # 必須在此映射，因 eval/test dump 均經 load_pair_sam_from_ablation 由
    # ablation_config.json 重建模型；缺此映射會靜默以 reference 模式評估。
    _prior_source = str(cfg.get('prior_source', 'reference'))
    if _prior_source not in ('reference', 'self'):
        raise ValueError(
            f"prior_source 必須為 'reference' 或 'self'，收到 {_prior_source!r}")
    model.prior_source = _prior_source
```

- [ ] **Step 5: 執行測試確認通過**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/test_prior_source_switch.py -v
```

Expected: 10 passed。

- [ ] **Step 6: 全套回歸測試**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/ -v --timeout=1200
```

Expected: 全部 passed。重點確認 `test_m2f_forward.py`、`test_deform_adapter_integration.py`、`test_build_from_config.py`、`test_sam_adapter_a3_api.py` 未因 forward 重構而失敗。

若 `--timeout` 參數不被支援（未安裝 pytest-timeout），去掉該旗標重跑。

- [ ] **Step 7: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/segment_anything/modeling/pair_sam.py \
        segment-anything/segment_anything/build_pair_sam.py \
        segment-anything/tests/test_prior_source_switch.py
git commit -m "feat(pair_sam): dispatch adapter prior on prior_source config

Extracts forward stage 0 into PairSAM._build_adapter_prior and adds a
prior_source attribute ('reference' default, 'self' for the same-image
baseline). The builder maps cfg['prior_source'] so eval and test-set dumps,
which rebuild from ablation_config.json, use the same prior path as training.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: CLI 旗標、config 落地與執行腳本

**Files:**
- Modify: `segment-anything/train.py`（argparse 約 331 行 `--adapter_variant` 之後；`abl_cfg` 約 356–369 行）
- Modify: `segment-anything/scripts/memcheck_m2f.py`（第 11 行 `cfg`）
- Create: `segment-anything/scripts/ablation_m2f_phase6_selfprior.sh`
- Test: 追加至 `segment-anything/tests/test_prior_source_switch.py`

**Interfaces:**
- Consumes: Task 2 的 `build_pair_sam_from_config` 對 `cfg['prior_source']` 的支援
- Produces: `train.py` 的 `--prior_source {reference,self}` 旗標（預設 `reference`），並將該值寫入 `abl_cfg`，經 `cfg_dump = {**abl_cfg, ...}` 自動落地至 `ablation_config.json`

- [ ] **Step 1: 追加失敗測試**

在 `segment-anything/tests/test_prior_source_switch.py` 末尾加入：

```python
def _train_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_train_cli_exposes_prior_source():
    """行為測試：--help 必須列出 --prior_source 及其兩個合法值。

    argparse parser 建構於 train.py 的 main() 內部、由 __main__ 守衛，
    無法 import 取得，故以子行程驗證真實的 argparse 行為。
    """
    import subprocess
    out = subprocess.run([sys.executable, 'train.py', '--help'],
                         cwd=_train_dir(), capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert '--prior_source' in out.stdout, "train.py 缺 --prior_source 旗標"
    assert '{reference,self}' in out.stdout, \
        "--prior_source 必須限定 choices=['reference', 'self']"


def test_train_cli_rejects_invalid_prior_source():
    """行為測試：非法值必須被 argparse 擋下，而非靜默落入預設。"""
    import subprocess
    out = subprocess.run([sys.executable, 'train.py', '--prior_source', 'bogus'],
                         cwd=_train_dir(), capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 2, f"預期 argparse 錯誤退出碼 2，實得 {out.returncode}"
    assert 'invalid choice' in out.stderr


def test_train_wires_prior_source_into_abl_cfg():
    """接線檢查：abl_cfg 必須含 prior_source=args.prior_source。

    這是原始碼層級斷言，不是行為測試 —— 刻意如此。要在執行期驗證需要
    完整的 CUDA 環境、資料集與 checkpoint，成本過高；真正的執行期驗證是
    Task 4 Step 2（訓練啟動兩分鐘後檢查 ablation_config.json）。

    保留本項的理由：若 abl_cfg 遺漏此鍵，eval 會從 ablation_config.json
    重建模型時靜默退回 reference 模式、產出錯誤數字。此斷言讓接線遺漏在
    改動 train.py 的當下就被發現，而不是八小時訓練跑完後才發現。
    """
    import re
    with open(os.path.join(_train_dir(), 'train.py'), encoding='utf-8') as f:
        src = f.read()
    assert re.search(r'prior_source\s*=\s*args\.prior_source', src), \
        "abl_cfg 必須含 prior_source=args.prior_source"
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/test_prior_source_switch.py -k prior_source_ -v
```

Expected: 三個新測試 FAIL —— `test_train_cli_exposes_prior_source` 報 `train.py 缺 --prior_source 旗標`、`test_train_cli_rejects_invalid_prior_source` 報退出碼非 2、`test_train_wires_prior_source_into_abl_cfg` 報 abl_cfg 缺鍵。

- [ ] **Step 3: 加入 CLI 旗標**

在 `segment-anything/train.py` 的 `--adapter_variant` 定義（331–333 行）之後加入：

```python
    parser.add_argument("--prior_source", choices=["reference", "self"], default="reference",
                        help="[P1 基線] DeformAdapter 先驗來源：reference=UAWarpC 對齊後的"
                             "跨視角晴天參考（FULL）；self=當前影像的 VGG 多尺度特徵"
                             "（ViT-Adapter 式 SPM 基線，不經對齊，conf≡1）")
```

在 `abl_cfg` 的 dict 中，於 `adapter_variant=args.adapter_variant,`（第 367 行）之後加入：

```python
        prior_source=args.prior_source,
```

`cfg_dump = {**abl_cfg, ...}`（第 380 行）會自動帶入，不需另外修改。

- [ ] **Step 4: 執行測試確認通過**

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python -m pytest segment-anything/tests/test_prior_source_switch.py -v
```

Expected: 13 passed。兩個子行程測試各需約 10–30 秒（載入 torch）。

- [ ] **Step 5: 讓 memcheck 支援 smoke**

在 `segment-anything/scripts/memcheck_m2f.py` 第 11 行 `cfg = {...}` 之前加入：

```python
import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--prior_source", choices=["reference", "self"], default="reference")
_args = _p.parse_args()
```

並將第 11 行改為：

```python
cfg = {"model_type": "vit_h", "use_vgg_adapter": True, "decoder": "m2f",
       "prior_source": _args.prior_source}
```

`self` 模式下 `batched` 仍帶 `clear_image`（模擬真實 dataloader，該鍵會被忽略），不需修改。

- [ ] **Step 6: 建立 Phase 6 執行腳本**

建立 `segment-anything/scripts/ablation_m2f_phase6_selfprior.sh`：

```bash
#!/usr/bin/env bash
# =============================================================================
# M-series Phase 6 — 同影像先驗基線（P1，約 8 小時）
#
# 目的：產出與完整模型只差單一變因的受控基線。P1 將 DeformAdapter 的先驗來源
#   由「UAWarpC 對齊後的跨視角晴天參考」換成「當前影像」（ViT-Adapter 式 SPM），
#   其餘主幹、解碼器、閘控零初始化、注入位置、抽取器、資料、排程、seed 全部相同。
#
# 回答兩個問題：
#   1. 先驗該取自跨視角參考還是當前影像？（對照 FULL_seed42 = 76.02）
#   2. W4_seed42 的 79.80 是否源自其固定 0.05 閘控初始化？P1 用零初始化閘控，
#      若 P1 亦顯著高於 FULL，則排除閘控假說。
#
# 設計文件：docs/superpowers/specs/2026-08-06-selfprior-baseline-design.md
#
# 假設已 conda activate sam_env。背景執行：
#   nohup bash scripts/ablation_m2f_phase6_selfprior.sh > outputs_ablation_m2f/phase6.log 2>&1 &
#
# 注意：本腳本只跑訓練與 ACDC val 評估。ACDC test 提交須另行執行
#   scripts/eval/dump_acdc_test_preds.py，且會消耗不可逆的官方 server 配額。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ablation_m2f_common.sh
mkdir -p "$OUT_ROOT"

run_one "P1_selfprior_seed42" --prior_source self

echo "===== Phase 6 完成 ====="
print_summary
```

設定執行權限：

```bash
chmod +x /home/rvl1421/SAM_research-1/segment-anything/scripts/ablation_m2f_phase6_selfprior.sh
```

- [ ] **Step 7: 驗證腳本會傳出正確旗標（不實跑訓練）**

```bash
cd /home/rvl1421/SAM_research-1/segment-anything
bash -n scripts/ablation_m2f_phase6_selfprior.sh && echo "SYNTAX OK"
conda run -n sam_env python train.py --help 2>&1 | grep -A3 "prior_source"
```

Expected: 印出 `SYNTAX OK`，且 help 顯示 `--prior_source {reference,self}`。

- [ ] **Step 8: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/train.py segment-anything/scripts/memcheck_m2f.py \
        segment-anything/tests/test_prior_source_switch.py
git add -f segment-anything/scripts/ablation_m2f_phase6_selfprior.sh
git commit -m "feat(train): add --prior_source flag and phase 6 ablation script

The flag lands in abl_cfg and therefore in ablation_config.json, which eval
and test-set dump read back to rebuild the model.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Smoke 驗證、訓練、ACDC val 評估與報告

**Files:**
- Create: `docs/experiments/2026-08-06-selfprior-baseline.md`
- 產出：`segment-anything/outputs_ablation_m2f/P1_selfprior_seed42/`

**Interfaces:**
- Consumes: Task 3 的 `scripts/ablation_m2f_phase6_selfprior.sh` 與 `--prior_source` 旗標
- Produces: `outputs_ablation_m2f/P1_selfprior_seed42/e1_results.json`（含 `overall_miou`）、`train_log.csv`、`ablation_config.json`、`weather_sam_best_latest.pth`

- [ ] **Step 1: Smoke —— 3 步訓練，確認可跑且顯存足夠**

```bash
cd /home/rvl1421/SAM_research-1/segment-anything
conda run -n sam_env python scripts/memcheck_m2f.py --prior_source self
```

Expected: 印出 3 行 `step N: loss=...`，loss 為有限值，最後印 `peak allocated: X.XX GiB` 與 `MEMCHECK PASS`，峰值須 ≤ 20 GiB。self 模式少一次 VGG 抽特徵並跳過 UAWarpC，峰值應不高於 reference 模式。

對照組（確認既有路徑未壞）：

```bash
conda run -n sam_env python scripts/memcheck_m2f.py --prior_source reference
```

Expected: 同樣 `MEMCHECK PASS`。

- [ ] **Step 2: 確認 `ablation_config.json` 會落地 `prior_source`**

先啟動訓練，等約 2 分鐘讓 config 寫出後檢查：

```bash
cd /home/rvl1421/SAM_research-1/segment-anything
mkdir -p outputs_ablation_m2f
nohup bash scripts/ablation_m2f_phase6_selfprior.sh \
    > outputs_ablation_m2f/phase6.log 2>&1 &
echo "PID=$!"
```

等待約 2 分鐘後：

```bash
cat outputs_ablation_m2f/P1_selfprior_seed42/ablation_config.json | python -m json.tool | grep prior_source
```

Expected: `"prior_source": "self"`。

**若此處不是 `self`，立刻 `kill` 訓練程序並回到 Task 3 修正** —— 繼續跑會浪費 8 小時並產出以 reference 模式評估的錯誤數字。

- [ ] **Step 3: 監看首個 epoch 的健康指標**

```bash
cd /home/rvl1421/SAM_research-1/segment-anything
tail -f outputs_ablation_m2f/phase6.log
```

檢查項目：

- loss 逐步下降，無 NaN/Inf
- 遙測 `conf_mean` 應為 `1.000`（self 模式的中性值）
- `[Grad Audit]` 若報 `vgg_injector` 未收到梯度即為異常 —— self 模式的 RPM `proj_c2/c3/c4` 應在計算圖上並收到梯度（與 `--no-ref` 不同，後者的良性警報源自 `c = zeros_like(c)`）

第 1 個 epoch 結束後（約 15 分鐘）確認：

```bash
head -2 outputs_ablation_m2f/P1_selfprior_seed42/train_log.csv | cut -d, -f1,13,29
```

Expected: `epoch,val_miou,train_inject_gate` 三欄有值，`train_inject_gate` 為接近 0 的小正值（零初始化閘控開始成長）。

- [ ] **Step 4: 等待訓練與評估完成**

腳本的 `run_one` 會在訓練後自動執行 `eval_e1_acdc_val_full.py`。全程約 7–9 小時。完成標誌：

```bash
cd /home/rvl1421/SAM_research-1/segment-anything
grep "Phase 6 完成" outputs_ablation_m2f/phase6.log
ls -la outputs_ablation_m2f/P1_selfprior_seed42/e1_results.json
```

- [ ] **Step 5: 擷取結果並與既有 run 對照**

```bash
cd /home/rvl1421/SAM_research-1/segment-anything/outputs_ablation_m2f
conda run -n sam_env python -c "
import json
for run in ['FULL_seed42','W2_semB_seed42','W4_seed42','P1_selfprior_seed42']:
    d = json.load(open(f'{run}/e1_results.json'))
    print(f\"{run:24s} overall={d['overall_miou']*100:.2f}\")
"
```

逐條件與逐類別（鍵名已由既有 `e1_results.json` 確認：`per_condition_miou` 為 `fog`/`rain`/`snow`/`night`，`per_class_iou_overall` 為 19 個 Cityscapes 類名）：

```bash
cd /home/rvl1421/SAM_research-1/segment-anything/outputs_ablation_m2f
conda run -n sam_env python -c "
import json
RUNS = ['FULL_seed42','W2_semB_seed42','W4_seed42','P1_selfprior_seed42']
data = {r: json.load(open(f'{r}/e1_results.json')) for r in RUNS}
conds = ['fog','rain','snow','night']
print(f\"{'run':24s} \" + ' '.join(f'{c:>7s}' for c in conds) + '  overall')
for r in RUNS:
    d = data[r]
    row = ' '.join(f\"{d['per_condition_miou'][c]*100:7.2f}\" for c in conds)
    print(f'{r:24s} ' + row + f\"  {d['overall_miou']*100:7.2f}\")
print()
classes = list(data['FULL_seed42']['per_class_iou_overall'].keys())
print(f\"{'class':16s} \" + ' '.join(f'{r.split(chr(95))[0]:>8s}' for r in RUNS))
for c in classes:
    print(f'{c:16s} ' + ' '.join(
        f\"{data[r]['per_class_iou_overall'][c]*100:8.2f}\" for r in RUNS))
"
```

閘控軌跡對照：

```bash
cd /home/rvl1421/SAM_research-1/segment-anything/outputs_ablation_m2f
conda run -n sam_env python -c "
import csv
for run in ['FULL_seed42','W4_seed42','P1_selfprior_seed42']:
    rows = list(csv.DictReader(open(f'{run}/train_log.csv')))
    best = max(rows, key=lambda r: float(r['val_miou']))
    print(f\"{run:24s} best_ep={best['epoch']:>2s} \"
          f\"train_log_peak={float(best['val_miou'])*100:.2f} \"
          f\"gate_final={float(rows[-1]['train_inject_gate']):.5f}\")
"
```

- [ ] **Step 6: 撰寫結果報告**

建立 `docs/experiments/2026-08-06-selfprior-baseline.md`，內容包含：

1. **Metadata**：run ID、config 摘要、best epoch、權重路徑
2. **主結果表**：P1 與 `FULL_seed42` / `W2_semB_seed42` / `W4_seed42` 的 `e1_results.json` `overall_miou` 對照，並標註每列與 FULL 的變因差異
3. **逐條件表**：霧／雨／雪／夜 四條件 mIoU
4. **逐類別表**：19 類 IoU
5. **閘控軌跡分析**：P1 與 FULL、W4 的 `train_inject_gate` 對比，回答「W4 的優勢是否源自閘控初始化」
6. **判讀**：明確陳述 P1 相對 FULL 的差值方向與幅度，對照 FULL 的種子標準差 0.14 判斷是否為穩定效應
7. **不做結論延伸** —— 是否納入論文由使用者決定

報告中不得寫入任何論文修改建議以外的既成事實；本 Task 不修改 `paper/` 之下任何檔案。

- [ ] **Step 7: Commit**

```bash
cd /home/rvl1421/SAM_research-1
git add -f docs/experiments/2026-08-06-selfprior-baseline.md
git commit -m "docs(experiments): report same-image prior baseline (P1) results

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: 停止並回報**

**在此停止。** 不執行 `scripts/eval/dump_acdc_test_preds.py`，不提交 ACDC 官方 server。

向使用者回報：P1 的 val 結果、與三個對照 run 的差值、閘控軌跡的判讀，以及三個後續選項（納入主比較表／置於消融章節／不納入但仍需修正論文 §4.5.2 對 W4 的收斂性解釋）。

ACDC test 提交會消耗不可逆的官方配額，須由使用者明確指示後才執行。屆時的指令為：

```bash
cd /home/rvl1421/SAM_research-1/segment-anything
conda run -n sam_env python scripts/eval/dump_acdc_test_preds.py \
    --csv ../Datasets/acdc_adverse_ref_rgb_test.csv \
    --ckpt outputs_ablation_m2f/P1_selfprior_seed42/weather_sam_best_latest.pth \
    --out  submissions/acdc_test_p1_selfprior \
    --zip
```

---

## 風險與注意事項

| 風險 | 徵兆 | 處置 |
|---|---|---|
| `ablation_config.json` 缺 `prior_source` | eval 靜默以 reference 模式重建模型 | Task 4 Step 2 為專門的檢查點，不通過即中止 |
| forward 重構破壞既有路徑 | `test_m2f_forward.py` 等失敗 | Task 2 Step 6 跑全套回歸 |
| self 模式誤觸 `_adapter_reference_free` | `vgg_injector` 無 `rpm` 屬性 | `test_self_mode_does_not_set_adapter_reference_free` 鎖定 |
| 閘控初始化不一致 | P1 與 FULL 多一個變因，結論失效 | `test_self_mode_keeps_zero_init_gate` 鎖定 |
| 拿 `train_log.csv` 峰值與既有 run 相比 | 產生假的 0.08 級差異 | Global Constraints 明列；Task 4 Step 5 兩個指令分開擷取 |
| P1 高於 Pair-SAM | — | 非技術風險。據實回報，由使用者決定處置（見設計文件 §10） |
