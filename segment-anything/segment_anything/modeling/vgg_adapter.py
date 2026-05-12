# segment-anything/segment_anything/modeling/vgg_adapter.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # ≈ -2.9444；softplus(x) ≈ 0.05


class MultiScaleCrossAttnInjector(nn.Module):
    """
    Multi-scale VGG Feature Injector（SAM-Adapter 風格，v4）。

    設計變更（相對 v3）：
      - 移除 Cross-Attention（Q 來自 ViT token → delta 被迫與 q 相似，inject_cos_sim 高）
      - 改用 SAM-Adapter 風格 MLP：VGG feats → pool → MLP_down/GELU/MLP_up → delta
        delta 完全獨立於 ViT token，inject_cos_sim 預期從 0.79 降至 0.5 以下
      - Gate：sigmoid(-5)≈0.007 → softplus(-2.94)≈0.05，梯度強 7×，上界 ∞
      - MLP_up zero-init：初期 delta=0，與 gate warmup 雙重保護訓練穩定性

    注入點：ViT-H Block [7, 15, 23, 31]（global attention blocks）
    輸入特徵：multi_scale_feats dict = {'l2': (B,256,H,W), 'l3': (B,512,H,W)}

    Diagnostics（trainer 相容）：
        _last_inject_cos_sim  : float — 4 stage 注入前後 cosine similarity 均值
        _last_gate_val        : float — 4 stage softplus(gate) 均值
        _last_delta_norm_ratio: float — inject_delta_norm / vit_token_norm
        _stage_cos_sims       : list[float] — per-stage cos_sim
        _stage_gate_vals      : list[float] — per-stage gate 值
    """

    INJECT_BLOCKS: list = [7, 15, 23, 31]

    def __init__(
        self,
        vit_dim: int = 1280,
        l2_channels: int = 256,
        l3_channels: int = 512,
        d_hidden: int = 256,
        pool_size: int = 32,
        gate_init: float = _DEFAULT_GATE_INIT,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.pool_size = pool_size
        kv_in = l2_channels + l3_channels  # 768

        num_stages = len(self.INJECT_BLOCKS)
        self._num_stages = num_stages

        # MLP_down: kv_in → d_hidden（per-stage）
        self.vgg_mlp_downs = nn.ModuleList([
            nn.Linear(kv_in, d_hidden) for _ in range(num_stages)
        ])

        # MLP_up: d_hidden → vit_dim（per-stage）；zero-init 避免初期擾動
        self.vgg_mlp_ups = nn.ModuleList([
            nn.Linear(d_hidden, vit_dim, bias=False) for _ in range(num_stages)
        ])
        for proj in self.vgg_mlp_ups:
            nn.init.zeros_(proj.weight)

        # Gate：softplus(raw_gate)，初始 ≈ 0.05
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
        q = output.reshape(B, H * W, C)

        # VGG feats → interpolate to (H,W) → concat → pool → flatten
        if f_l2.shape[-2:] != (H, W):
            f_l2 = F.interpolate(f_l2, size=(H, W), mode='bilinear', align_corners=False)
        if f_l3.shape[-2:] != (H, W):
            f_l3 = F.interpolate(f_l3, size=(H, W), mode='bilinear', align_corners=False)

        f_concat = torch.cat([f_l2, f_l3], dim=1)                          # (B, kv_in, H, W)
        f_pooled = F.adaptive_avg_pool2d(f_concat, (self.pool_size, self.pool_size))
        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, self.pool_size ** 2, -1)  # (B, P², kv_in)

        # MLP: (B, P², kv_in) → (B, P², d_hidden) → (B, P², vit_dim)
        h = F.gelu(self.vgg_mlp_downs[stage_idx](f_flat))
        delta_spatial = self.vgg_mlp_ups[stage_idx](h)                     # (B, P², vit_dim)

        # Bilinear upsample: pool_size×pool_size → H×W，保留空間結構
        delta_map = delta_spatial.reshape(B, self.pool_size, self.pool_size, self.vit_dim)
        delta_map = delta_map.permute(0, 3, 1, 2)                          # (B, vit_dim, P, P)
        delta_map = F.interpolate(delta_map, size=(H, W), mode='bilinear', align_corners=False)
        delta = delta_map.permute(0, 2, 3, 1).reshape(B, H * W, self.vit_dim)  # (B, H*W, vit_dim)

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
