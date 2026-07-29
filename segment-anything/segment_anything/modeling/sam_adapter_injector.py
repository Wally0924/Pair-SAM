# segment-anything/segment_anything/modeling/sam_adapter_injector.py
"""SAM-Adapter 風格的基線注入器（實驗 B）。

設計目的：作為「跨視角參考注入」的對照基線。它鏡像 MultiScaleCrossAttnInjector
的 hook 介面（相同 INJECT_BLOCKS / set_features / _make_pre_hook / .gates / 診斷
欄位），讓 PairSAM 能透過完全相同的 enable/disable 路徑在相同的 ViT-H 區塊註冊
hook，並沿用既有的 optimizer 參數分組與 gate warmup。

與 MultiScaleCrossAttnInjector 的唯一差異：殘差 delta 改由「對 ViT token 本身做
per-block bottleneck MLP（down -> GELU -> up）」產生，屬同影像（same-image）adapter，
因此模型不接收任何跨視角參考資訊。藉此量測「跨視角參考」相對於標準同影像 adapter
在相同凍結骨幹設定下多帶來多少效益。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # softplus(x) ≈ 0.05


class SameImageAdapterInjector(nn.Module):
    INJECT_BLOCKS: list = [7, 15, 23, 31]
    EXTRACT_BLOCKS: list = []   # 基線無 extractor;A3 enable_vgg_adapter 的 extract 迴圈自然零次跳過

    def __init__(self, vit_dim: int = 1280, bottleneck: int = 1700,
                 gate_init: float = _DEFAULT_GATE_INIT):
        super().__init__()
        self.vit_dim = vit_dim
        num_stages = len(self.INJECT_BLOCKS)
        self._num_stages = num_stages

        # 每注入點一個 bottleneck adapter：down -> GELU -> up（SAM-Adapter 風格）
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vit_dim, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, vit_dim),
            )
            for _ in range(num_stages)
        ])
        # up-projection 用小 std 初始化（比照參考注入器「不用 zero-init 以避免
        # softplus gate 下梯度死鎖」的策略），讓初期 delta 微小但非零。
        for adp in self.adapters:
            nn.init.normal_(adp[-1].weight, std=0.01)
            nn.init.zeros_(adp[-1].bias)

        # Gate：softplus(raw)，init ≈ 0.05，無上界（與參考注入器一致，trainer 可凍結）
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            for _ in range(num_stages)
        ])

        # 診斷欄位（與 MultiScaleCrossAttnInjector 對齊，供 trainer 讀取）
        _init_gate = float(F.softplus(torch.tensor(gate_init)))
        self._last_inject_cos_sim: float = 1.0
        self._last_gate_val: float = _init_gate
        self._last_delta_norm_ratio: float = 0.0
        self._last_kv_keep_ratio: float = 1.0      # 永遠 1.0（無 masking）；保留供 trainer 相容
        self._stage_cos_sims: list = [1.0] * num_stages
        self._stage_gate_vals: list = [_init_gate] * num_stages
        self._global_step: int = 0
        self._stages_fired: int = 0
        self.use_reference: bool = False           # 基線永不使用參考；保留以維持 API 一致

    def set_features(self, multi_scale_feats: dict):
        """接收後忽略：與參考注入器的呼叫端介面一致（forward 對基線其實不會呼叫）。"""
        self._stages_fired = 0

    def _make_pre_hook(self, stage_idx: int):
        def hook(module, input):
            return (self._inject_at_stage(input[0], stage_idx),)
        return hook

    def _make_hook(self, stage_idx: int):
        def hook(module, input, output):
            return self._inject_at_stage(output, stage_idx)
        return hook

    # A3 hook API 別名:enable_vgg_adapter(DeformAdapter 介面)呼叫此名稱。
    # 注入語意不變(pre-hook 殘差);_inject_at_stage 為 token 的純函數,
    # gradient checkpointing 重放安全,無需 DeformAdapter 的快照機制。
    _make_inject_pre_hook = _make_pre_hook

    def _inject_at_stage(self, output: torch.Tensor, stage_idx: int) -> torch.Tensor:
        B, H, W, C = output.shape
        q = output.reshape(B, H * W, C)
        delta = self.adapters[stage_idx](q)            # 同影像 bottleneck adapter
        gate = F.softplus(self.gates[stage_idx])
        injected_flat = q + gate * delta
        injected = injected_flat.reshape(B, H, W, C)

        with torch.no_grad():
            cos = F.cosine_similarity(q, injected_flat, dim=-1).mean().item()
            self._stage_cos_sims[stage_idx] = cos
            self._stage_gate_vals[stage_idx] = float(gate.item())
            if stage_idx == 0:
                delta_norm = (gate * delta).norm(dim=-1).mean().item()
                vit_norm = q.norm(dim=-1).mean().item()
                self._last_delta_norm_ratio = delta_norm / (vit_norm + 1e-8)

        self._stages_fired += 1
        if self._stages_fired >= self._num_stages:
            self._last_inject_cos_sim = float(sum(self._stage_cos_sims) / self._num_stages)
            self._last_gate_val = float(sum(self._stage_gate_vals) / self._num_stages)
            self._global_step += 1
            self._stages_fired = 0

        return injected
