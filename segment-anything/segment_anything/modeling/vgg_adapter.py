# segment-anything/segment_anything/modeling/vgg_adapter.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # ≈ -2.9444；softplus(x) ≈ 0.05


class MultiScaleCrossAttnInjector(nn.Module):
    """
    Multi-scale VGG Feature Injector（Cross-Attention v5.1，confidence-aware）。

    設計：
      - Q = ViT token（.detach()，stop_gradient）— 場景感知但梯度不回傳至 ViT
      - K, V = VGG feats（l2+l3 concat → weighted pool → Linear(kv_in, d_attn)）
      - MHA(embed_dim=vit_dim, kdim=d_attn, vdim=d_attn, num_heads, batch_first=True)
      - **key_padding_mask**：valid_ratio < mask_threshold 的 pooled cell 從 attention 中排除
      - 所有投影 Xavier init，無 zero-init 梯度死鎖
      - Gate: softplus(init≈0.05) + trainer gate warmup 保護穩定性

    Confidence-aware 處理（相對 v5 主結構的補強）：
      - VGG l2/l3 已在 fusion.pre_align 中乘過 hard_mask（conf < 0.2 歸零）
      - 但 average pool 會稀釋有效特徵，且 attention 仍會 attend 到零位置
      - 修正：(1) weighted pool = avg(f*m) / avg(m)，消除零位置稀釋
              (2) key_padding_mask：valid_ratio < mask_threshold 的 cell 從 attention 排除
      - 若 set_features 提供的 dict 沒有 'mask' 鍵，行為退回 v5 原始邏輯（向後相容）

    注入點：ViT-H Block [7, 15, 23, 31]（global attention blocks）
    輸入特徵：multi_scale_feats dict = {'l2': (B,256,H,W),
                                       'l3': (B,512,H,W),
                                       'mask': (B,1,H,W)  # 可選；fusion.pre_align 提供}

    Diagnostics（trainer 相容）：
        _last_inject_cos_sim  : float — cos(q, injected) 4 stage 均值
        _last_gate_val        : float — softplus(gate) 4 stage 均值
        _last_delta_norm_ratio: float — ||gate*delta|| / ||q||
        _stage_cos_sims       : list[float] — per-stage cos_sim
        _stage_gate_vals      : list[float] — per-stage gate 值
    """

    INJECT_BLOCKS: list = [7, 15, 23, 31]

    def __init__(
        self,
        vit_dim: int = 1280,
        l2_channels: int = 256,
        l3_channels: int = 512,
        d_attn: int = 256,
        pool_size: int = 32,
        num_heads: int = 4,
        mask_threshold: float = 0.3,
        gate_init: float = _DEFAULT_GATE_INIT,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.pool_size = pool_size
        self.mask_threshold = mask_threshold
        kv_in = l2_channels + l3_channels  # 768

        num_stages = len(self.INJECT_BLOCKS)
        self._num_stages = num_stages

        # K/V projections：VGG concat → d_attn（Xavier init by default）
        self.k_projs = nn.ModuleList([
            nn.Linear(kv_in, d_attn) for _ in range(num_stages)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(kv_in, d_attn) for _ in range(num_stages)
        ])

        # Cross-attention：Q(vit_dim) × K(d_attn) × V(d_attn) → delta(vit_dim)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=vit_dim,
                kdim=d_attn,
                vdim=d_attn,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_stages)
        ])

        # Gate：softplus(raw_gate)，初始 ≈ 0.05，無上界
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            for _ in range(num_stages)
        ])

        self._multi_scale_feats: dict = None
        self._stages_fired: int = 0

        _init_gate = float(F.softplus(torch.tensor(gate_init)))
        self._last_inject_cos_sim: float = 1.0
        self._last_gate_val: float = _init_gate
        self._last_delta_norm_ratio: float = 0.0
        self._last_kv_keep_ratio: float = 1.0   # valid cells / total cells（≤1，越低代表 mask 過濾越多）
        self._stage_cos_sims: list = [1.0] * num_stages
        self._stage_gate_vals: list = [_init_gate] * num_stages
        self._global_step: int = 0

    def set_features(self, multi_scale_feats: dict):
        self._multi_scale_feats = multi_scale_feats
        self._stages_fired = 0

    def _make_hook(self, stage_idx: int):
        def hook(module, input, output):
            return self._inject_at_stage(output, stage_idx)
        return hook

    def _make_pre_hook(self, stage_idx: int):
        def hook(module, input):
            return (self._inject_at_stage(input[0], stage_idx),)
        return hook

    def _inject_at_stage(self, output: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if self._multi_scale_feats is None:
            return output

        f_l2 = self._multi_scale_feats['l2'].to(output.device, dtype=output.dtype)
        f_l3 = self._multi_scale_feats['l3'].to(output.device, dtype=output.dtype)
        m_raw = self._multi_scale_feats.get('mask', None)   # (B,1,H,W) 或 None（向後相容）
        if m_raw is not None:
            m_raw = m_raw.to(output.device, dtype=output.dtype)

        B, H, W, C = output.shape
        q = output.reshape(B, H * W, C)   # 有梯度，用於殘差加法
        Q = q.detach()                     # 無梯度，用於 attention key selection

        # VGG feats → interpolate to (H,W) → concat
        if f_l2.shape[-2:] != (H, W):
            f_l2 = F.interpolate(f_l2, size=(H, W), mode='bilinear', align_corners=False)
        if f_l3.shape[-2:] != (H, W):
            f_l3 = F.interpolate(f_l3, size=(H, W), mode='bilinear', align_corners=False)
        f_concat = torch.cat([f_l2, f_l3], dim=1)  # (B, kv_in, H, W)

        P = self.pool_size
        if m_raw is not None:
            # ── Mask-aware path：weighted pool + key_padding_mask ──
            if m_raw.shape[-2:] != (H, W):
                m_raw = F.interpolate(m_raw, size=(H, W), mode='nearest')

            # 分別 pool 特徵與 mask：
            #   avg_pool(f*m) / avg_pool(m) = sum(f*m)/sum(m) = 有效 cell 的加權平均
            #   注意：f_concat 已在 fusion.pre_align 中乘過 hard_mask（confidence<0.2 歸零），
            #         所以 f_concat = f_raw * m_raw；avg_pool(f_concat) 即 avg_pool(f*m)
            f_pooled_num = F.adaptive_avg_pool2d(f_concat, (P, P))     # (B, kv_in, P, P)
            m_pooled     = F.adaptive_avg_pool2d(m_raw,    (P, P))     # (B, 1,     P, P) in [0,1]
            f_pooled     = f_pooled_num / (m_pooled + 1e-6)            # weighted avg

            # key_padding_mask：True = 忽略；valid_ratio < threshold 的 cell 被排除
            m_flat = m_pooled.flatten(2).squeeze(1)                    # (B, P²)
            key_padding_mask = (m_flat < self.mask_threshold)          # (B, P²) bool

            # 安全防護：若某 sample 全部 cell 被排除，會造成 softmax NaN
            #   → 強制保留 valid_ratio 最大的 cell
            row_all_masked = key_padding_mask.all(dim=1)               # (B,)
            if row_all_masked.any():
                fallback_idx = m_flat.argmax(dim=1)                    # (B,)
                batch_idx    = torch.arange(B, device=m_flat.device)
                rows_need    = batch_idx[row_all_masked]
                cols_need    = fallback_idx[row_all_masked]
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[rows_need, cols_need] = False

            # 診斷：保留比例（越低代表 mask 過濾越多）
            with torch.no_grad():
                keep_ratio = (~key_padding_mask).float().mean().item()
                if stage_idx == 0:
                    self._last_kv_keep_ratio = float(keep_ratio)
        else:
            # ── 向後相容：無 mask 時退回 v5 原始邏輯 ──
            f_pooled = F.adaptive_avg_pool2d(f_concat, (P, P))
            key_padding_mask = None

        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, P * P, -1)    # (B, P², kv_in)

        # K, V projections（Xavier init）
        K = self.k_projs[stage_idx](f_flat)   # (B, P², d_attn)
        V = self.v_projs[stage_idx](f_flat)   # (B, P², d_attn)

        # Cross-attention：Q(detach) × K × V → delta
        # need_weights=False 讓 PyTorch 走 SDPA / flash attention 快速路徑，
        # 不物化 (B, heads, L_q, L_k) 的 attention matrix（~250 MB），記憶體大幅下降
        delta, _ = self.cross_attns[stage_idx](
            Q, K, V,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, H*W, vit_dim)

        gate = F.softplus(self.gates[stage_idx])
        injected_flat = q + gate * delta
        injected = injected_flat.reshape(B, H, W, C)

        with torch.no_grad():
            cos = F.cosine_similarity(q, injected_flat, dim=-1).mean().item()
            self._stage_cos_sims[stage_idx] = cos
            self._stage_gate_vals[stage_idx] = float(gate.item())
            if stage_idx == 0:
                delta_norm = (gate * delta).norm(dim=-1).mean().item()
                vit_norm   = q.norm(dim=-1).mean().item()
                self._last_delta_norm_ratio = delta_norm / (vit_norm + 1e-8)

        self._stages_fired += 1
        if self._stages_fired >= self._num_stages:
            self._last_inject_cos_sim = float(sum(self._stage_cos_sims) / self._num_stages)
            self._last_gate_val = float(sum(self._stage_gate_vals) / self._num_stages)
            self._global_step += 1
            self._multi_scale_feats = None
            self._stages_fired = 0

        return injected
