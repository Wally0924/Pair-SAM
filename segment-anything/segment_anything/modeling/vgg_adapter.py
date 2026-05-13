# segment-anything/segment_anything/modeling/vgg_adapter.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # ≈ -2.9444；softplus(x) ≈ 0.05


class MultiScaleCrossAttnInjector(nn.Module):
    """
    Multi-scale VGG Feature Injector（Cross-Attention v5）。

    設計：
      - Q = ViT token（.detach()，stop_gradient）— 場景感知但梯度不回傳至 ViT
      - K, V = VGG feats（l2+l3 concat → pool(pool_size²) → Linear(kv_in, d_attn)）
      - MHA(embed_dim=vit_dim, kdim=d_attn, vdim=d_attn, num_heads, batch_first=True)
      - 所有投影 Xavier init（PyTorch MHA/Linear 預設），無 zero-init 梯度死鎖
      - Gate: softplus(init≈0.05) + trainer gate warmup 保護穩定性

    相對 v4（MLP）的改動：
      - 移除 vgg_mlp_downs / vgg_mlp_ups（MLP 路徑）
      - 新增 k_projs / v_projs / cross_attns（Cross-Attention 路徑）
      - Q 不經瓶頸壓縮，保留完整 vit_dim 語意

    注入點：ViT-H Block [7, 15, 23, 31]（global attention blocks）
    輸入特徵：multi_scale_feats dict = {'l2': (B,256,H,W), 'l3': (B,512,H,W)}

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
        gate_init: float = _DEFAULT_GATE_INIT,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.pool_size = pool_size
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

        B, H, W, C = output.shape
        q = output.reshape(B, H * W, C)   # 有梯度，用於殘差加法
        Q = q.detach()                     # 無梯度，用於 attention key selection

        # VGG feats → interpolate to (H,W) → concat → pool → flatten
        if f_l2.shape[-2:] != (H, W):
            f_l2 = F.interpolate(f_l2, size=(H, W), mode='bilinear', align_corners=False)
        if f_l3.shape[-2:] != (H, W):
            f_l3 = F.interpolate(f_l3, size=(H, W), mode='bilinear', align_corners=False)

        f_concat = torch.cat([f_l2, f_l3], dim=1)  # (B, kv_in, H, W)
        f_pooled = F.adaptive_avg_pool2d(f_concat, (self.pool_size, self.pool_size))
        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, self.pool_size ** 2, -1)  # (B, P², kv_in)

        # K, V projections（Xavier init）
        K = self.k_projs[stage_idx](f_flat)   # (B, P², d_attn)
        V = self.v_projs[stage_idx](f_flat)   # (B, P², d_attn)

        # Cross-attention：Q(detach) × K × V → delta
        delta, _ = self.cross_attns[stage_idx](Q, K, V)  # (B, H*W, vit_dim)

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
