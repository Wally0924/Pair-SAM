# vgg_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCrossAttnInjector(nn.Module):
    """
    Multi-scale Bottleneck Cross-Attention Adapter。

    設計要點（v3）：
      - Q 先壓縮到 d_attn=256（瓶頸），attention 在小維度計算，避免 embed_dim=1280 的 Q
        projection（1.6M/stage）造成 ACDC 1,200 張訓練集過擬合。
      - K/V 分開投影（k_proj/v_proj），各自學習不同的特徵編碼。
      - q_up_proj：xavier 初始（非零），gate 使用 -5.0（sigmoid≈0.007）補償初期擾動。
      - 無 zero-init 的投影層阻斷梯度路徑，gate 從第一步即可學習。

    參數量：~920K/stage，4 stages 合計 ~3.7M（舊版 MultiStageWarpedVGGInjector 的 1.6×）。

    注入點：ViT-H Block [7, 15, 23, 31]（global attention blocks）
    輸入特徵：multi_scale_feats dict = {'l2': (B,256,H,W), 'l3': (B,512,H,W)}

    Diagnostics（trainer 兼容）：
        _last_inject_cos_sim  : float — 4 stage 注入前後 cosine similarity 均值
        _last_gate_val        : float — 4 stage sigmoid(gate) 均值
        _last_delta_norm_ratio: float — inject_delta_norm / vit_token_norm（訓練初期監控）
    """

    INJECT_BLOCKS: list = [7, 15, 23, 31]  # ViT-H global attention blocks

    def __init__(
        self,
        vit_dim: int = 1280,
        d_attn: int = 256,       # Q 瓶頸維度（決定 attention 計算規模）
        l2_channels: int = 256,
        l3_channels: int = 512,
        d_kv: int = 64,          # K/V 投影維度；num_heads=4, head_dim=d_attn/4=64 ≡ d_kv
        pool_size: int = 32,
        num_heads: int = 4,
        gate_init: float = -5.0, # sigmoid(-5) ≈ 0.007；q_up_proj 非零，需更保守初始
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.d_attn = d_attn
        self.pool_size = pool_size
        kv_in_channels = l2_channels + l3_channels  # 768

        num_stages = len(self.INJECT_BLOCKS)
        self._num_stages = num_stages

        # Q 瓶頸壓縮（1280 → d_attn=256）
        self.q_down_projs = nn.ModuleList([
            nn.Linear(vit_dim, d_attn)
            for _ in range(num_stages)
        ])

        # K/V 分開投影（768 → d_kv=64）
        self.k_projs = nn.ModuleList([
            nn.Linear(kv_in_channels, d_kv)
            for _ in range(num_stages)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(kv_in_channels, d_kv)
            for _ in range(num_stages)
        ])

        # 瓶頸 Cross-Attention（embed_dim=d_attn，在小維度計算）
        # MHA 內建 W_o (xavier_uniform_)，attn_out 非零，gate 梯度正常
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_attn,
                num_heads=num_heads,
                kdim=d_kv,
                vdim=d_kv,
                batch_first=True,
                dropout=0.0,
            )
            for _ in range(num_stages)
        ])

        # Q 瓶頸擴張（d_attn=256 → vit_dim=1280）— xavier 初始（非零）
        # 不使用零初始化，避免 gate 梯度斷路；初期擾動由 gate_init=-5.0 控制
        self.q_up_projs = nn.ModuleList([
            nn.Linear(d_attn, vit_dim, bias=False)
            for _ in range(num_stages)
        ])
        for proj in self.q_up_projs:
            nn.init.xavier_uniform_(proj.weight)

        # Gate（初始值 sigmoid(-5.0) ≈ 0.007；比原版更保守，適配非零 q_up_proj）
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(gate_init)) for _ in range(num_stages)
        ])

        self._multi_scale_feats: dict = None
        self._stages_fired: int = 0

        _init_gate = float(torch.sigmoid(torch.tensor(gate_init)))
        self._last_inject_cos_sim: float = 1.0
        self._last_gate_val: float = _init_gate
        self._last_delta_norm_ratio: float = 0.0  # inject_delta_norm / vit_token_norm
        self._stage_cos_sims: list = [1.0] * num_stages
        self._stage_gate_vals: list = [_init_gate] * num_stages
        self._global_step: int = 0  # 供早期訓練監控使用

    def set_features(self, multi_scale_feats: dict):
        """在每次 WeatherSAM.forward 呼叫 image_encoder 前設定多尺度對齊特徵。

        Args:
            multi_scale_feats: dict with keys 'l2' (B,256,H,W) and 'l3' (B,512,H,W)
        """
        self._multi_scale_feats = multi_scale_feats
        self._stages_fired = 0

    def _make_hook(self, stage_idx: int):
        """為指定 stage 建立 forward hook closure，正確捕捉 stage_idx。"""
        def hook(module, input, output):
            return self._inject_at_stage(output, stage_idx)
        return hook

    def _make_pre_hook(self, stage_idx: int):
        """Pre-hook fires before block's self-attn; compensation participates in Q/K/V."""
        def hook(module, input):
            return (self._inject_at_stage(input[0], stage_idx),)
        return hook

    def _inject_at_stage(self, output: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """
        在指定 stage 執行瓶頸式 Cross-Attention 注入。

        output shape（ViT Block 輸出）：(B, H, W, C) = (B, 64, 64, 1280)
        """
        if self._multi_scale_feats is None:
            return output

        f_l2 = self._multi_scale_feats['l2'].to(output.device, dtype=output.dtype)
        f_l3 = self._multi_scale_feats['l3'].to(output.device, dtype=output.dtype)

        B, H, W, C = output.shape  # H=W=64, C=1280

        # ── Q：ViT tokens reshape + 瓶頸壓縮 ──
        q = output.reshape(B, H * W, C)                        # (B, 4096, 1280)
        q_down = self.q_down_projs[stage_idx](q)               # (B, 4096, 256)

        # ── KV：多尺度 VGG → pool → K/V 分開投影 ──
        if f_l2.shape[-2:] != (H, W):
            f_l2 = F.interpolate(f_l2, size=(H, W), mode='bilinear', align_corners=False)
        if f_l3.shape[-2:] != (H, W):
            f_l3 = F.interpolate(f_l3, size=(H, W), mode='bilinear', align_corners=False)

        f_concat = torch.cat([f_l2, f_l3], dim=1)              # (B, 768, H, W)
        f_pooled = F.adaptive_avg_pool2d(f_concat, (self.pool_size, self.pool_size))
        N_kv = self.pool_size * self.pool_size
        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, N_kv, -1)  # (B, 1024, 768)

        k = self.k_projs[stage_idx](f_flat)   # (B, 1024, 64)
        v = self.v_projs[stage_idx](f_flat)   # (B, 1024, 64)

        # ── 瓶頸 Cross-Attention（強制 fp32 避免 AMP fp16 overflow）──
        # AMP autocast 下 MHA 的 Q@K^T 在 fp16 可能 overflow（>65504），
        # 且 GradScaler init_scale=8192 時 actual_grad>8.0 即觸發 inf 鏈。
        # 強制 fp32 後 attn_out 轉回原 dtype，梯度路徑不受影響。
        with torch.amp.autocast('cuda', enabled=False):
            attn_out, _ = self.cross_attns[stage_idx](
                query=q_down.float(),   # (B, 4096, 256)
                key=k.float(),          # (B, 1024, 64)
                value=v.float(),        # (B, 1024, 64)
                need_weights=False,
            )  # attn_out: (B, 4096, 256) in fp32
        attn_out = attn_out.to(q_down.dtype)  # 轉回 fp16（若 autocast 啟用）

        # ── Q 擴張 + gate + residual ──
        delta = self.q_up_projs[stage_idx](attn_out)   # (B, 4096, 1280)
        gate = torch.sigmoid(self.gates[stage_idx])
        injected_flat = q + gate * delta               # (B, 4096, 1280)
        injected = injected_flat.reshape(B, H, W, C)

        # 診斷指標（含早期訓練監控）
        with torch.no_grad():
            cos = F.cosine_similarity(q, injected_flat, dim=-1).mean().item()
            self._stage_cos_sims[stage_idx] = cos
            self._stage_gate_vals[stage_idx] = float(gate.item())

            if stage_idx == 0:  # 只在 stage 0 計算 delta_norm_ratio，避免多次重複
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
