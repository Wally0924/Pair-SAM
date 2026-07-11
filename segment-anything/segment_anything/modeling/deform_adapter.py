"""WeatherSAM 雙向可變形 Adapter（A3）。SPM → UAWarpC 參考；Injector + Extractor。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops.ms_deform_attn import MSDeformAttn


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


class ReferencePriorModule(nn.Module):
    """取代 ViT-Adapter SPM：把 UAWarpC 對齊的 VGG 參考轉成 3 尺度 token 流。
    1/8 ← l2（conv3）；1/16 ← l3（conv4）；1/32 ← l4（conv5，真 stride-32）。"""
    def __init__(self, l2_channels=256, l3_channels=512, l4_channels=512,
                 dim=1280, use_reference=True):
        super().__init__()
        self.dim = dim
        self.use_reference = use_reference
        # W3 消融開關：False = 移除置信度調變（m̄≡1，參考特徵不分可靠與否全幅注入）。
        # 由 build_weather_sam_from_config 依 cfg['conf_mod'] 覆蓋，預設不影響既有行為。
        self.use_conf_mod = True
        self.proj_c2 = nn.Conv2d(l2_channels, dim, kernel_size=1)
        self.proj_c3 = nn.Conv2d(l3_channels, dim, kernel_size=1)
        self.proj_c4 = nn.Conv2d(l4_channels, dim, kernel_size=1)
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(self, feats):
        l2 = feats['l2']; l3 = feats['l3']; l4 = feats['l4']
        B = l2.shape[0]
        c2 = self.proj_c2(l2)                    # (B,dim,H8,W8)
        c3 = self.proj_c3(l3)                    # (B,dim,H16,W16)
        c4 = self.proj_c4(l4)                    # (B,dim,H32,W32)

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
        if mask is not None and self.use_reference and self.use_conf_mod:
            m2 = F.adaptive_avg_pool2d(mask, c2.shape[-2:])
            m3 = F.adaptive_avg_pool2d(mask, c3.shape[-2:])
            m4 = F.adaptive_avg_pool2d(mask, c4.shape[-2:])
            conf = torch.cat([_flat(m2), _flat(m3), _flat(m4)], dim=1)  # (B,L,1)
        else:
            # 無參考消融（use_reference=False）或移除置信度調變（use_conf_mod=False，W3：
            # m̄≡1、參考特徵全幅注入）：conf 設中性值 1，而非全零。前者參考特徵 c 已歸零，
            # 但 Adapter 仍照常運作（靠 extractor 回收骨幹語境自我精修）。若把 conf 歸零會乘性
            # 抹除注入、使 Adapter 近乎惰性，反而混入「移除 Adapter 容量」的效果；設為 1 則保留
            # 容量、只移除參考影像所提供之資訊（對齊特徵與其置信度加權），使「+參考」列與本列
            # 相減可乾淨歸因參考的淨貢獻（見論文 §4.5）。
            conf = torch.ones(B, c.shape[1], 1, device=c.device, dtype=c.dtype)
        return c, conf


class Injector(nn.Module):
    """ViT-Adapter Injector（可變形）。Q=ViT（端到端梯度回流），K/V=多尺度 c（信心加權）。
    gamma 依原論文 per-channel 零初始化：初始為恆等映射，不擾動 SAM 預訓練分布。"""
    def __init__(self, dim=1280, n_heads=8, n_points=4, n_levels=3,
                 deform_ratio=0.5):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=n_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x_tokens, c, conf, inject_inputs):
        ref_pts, spatial_shapes, lsi = inject_inputs
        ref_pts = ref_pts.to(x_tokens.device)
        if ref_pts.shape[0] != x_tokens.shape[0]:
            ref_pts = ref_pts.expand(x_tokens.shape[0], -1, -1, -1)
        q = self.query_norm(x_tokens)                    # 端到端：Q 梯度回流至 ViT stream（ViT-Adapter 設計）
        feat = self.feat_norm(c) * conf                  # 信心加權 value（決策③）
        delta = self.attn(q, ref_pts, feat, spatial_shapes, lsi)
        return x_tokens + self.gamma * delta             # 殘差 + Q 兩條路徑皆回流 ViT stream


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
    """ViT-Adapter Extractor（可變形）。Q=c，K/V=ViT（端到端梯度回流），+ ConvFFN 逐尺度精修 c。"""
    def __init__(self, dim=1280, n_heads=8, n_points=4, deform_ratio=0.5,
                 with_cffn=True):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=1, n_heads=n_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(dim=dim)
            self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, c, x_tokens, extract_inputs, scale_hw):
        ref_pts, spatial_shapes, lsi = extract_inputs
        ref_pts = ref_pts.to(c.device)
        if ref_pts.shape[0] != c.shape[0]:
            ref_pts = ref_pts.expand(c.shape[0], -1, -1, -1)
        feat = self.feat_norm(x_tokens)                  # 端到端：K/V 梯度回流至 ViT stream（ViT-Adapter 設計）
        attn = self.attn(self.query_norm(c), ref_pts, feat, spatial_shapes, lsi)
        c = c + attn
        if self.with_cffn:
            c = c + self.ffn(self.ffn_norm(c), scale_hw)
        return c


class DeformAdapter(nn.Module):
    """雙向可變形 adapter 協調器：管理多尺度 c 狀態、4 injector + 3 extractor、hook 工廠。
    非侵入：透過 ViT block 的 forward pre/post hook 加殘差，不改 encoder 結構。"""
    INJECT_BLOCKS = [0, 8, 16, 24]
    EXTRACT_BLOCKS = [7, 15, 23]

    def __init__(self, vit_dim=1280, l2_channels=256, l3_channels=512,
                 l4_channels=512, n_heads=8, deform_ratio=0.5, use_reference=True):
        super().__init__()
        self.rpm = ReferencePriorModule(l2_channels, l3_channels, l4_channels,
                                        dim=vit_dim, use_reference=use_reference)
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
        # gradient checkpointing 會在 backward 重放 hook；重放時 self._c 已被後續
        # extractor 改寫，直接讀會使重算 forward 偏離原 forward → 梯度錯誤。
        # 因此每 stage 首次觸發時快照當下的 c，重放一律讀快照（逐位元重現）。
        self._inject_c = [None] * len(self.INJECT_BLOCKS)
        self._extract_c = [None] * len(self.EXTRACT_BLOCKS)
        # telemetry（trainer 讀取；介面與 legacy adapter 對齊）
        self._num_stages = len(self.INJECT_BLOCKS)
        self._stage_gate_vals = [0.0] * self._num_stages   # gamma.abs().mean() per stage
        self._stage_cos_sims = [1.0] * self._num_stages
        self._last_gate_val = 0.0            # gamma.abs().mean()：零初始化起步
        self._last_inject_cos_sim = 1.0
        # ||注入殘差(out-in)|| / ||ViT token||：實際注入幅度比值。與 sam_adapter 的
        # (gate*delta)/q 同語意，補上此欄位讓 trainer 的 hasattr 守門不再靜默跳過。
        self._last_delta_norm_ratio = 0.0

    def set_features(self, feats, h, w):
        device = feats['l2'].device
        self._c, self._conf = self.rpm(feats)
        self._inject_inputs, self._extract_inputs = deform_inputs(h, w, device)
        self._scale_hw = [(h * 2, w * 2), (h, w), (h // 2, w // 2)]
        self._inject_c = [None] * len(self.INJECT_BLOCKS)   # 新 forward → 快照重置
        self._extract_c = [None] * len(self.EXTRACT_BLOCKS)

    def _make_inject_pre_hook(self, stage_idx):
        def hook(module, inp):
            if self._c is None:
                return None                      # 未 set_features：原樣放行
            x = inp[0]
            B, H, W, C = x.shape
            tokens = x.reshape(B, H * W, C)
            if self._inject_c[stage_idx] is None:
                self._inject_c[stage_idx] = self._c        # 首次觸發：快照
            c_in = self._inject_c[stage_idx]               # 重放讀快照，非現時 _c
            out = self.injectors[stage_idx](tokens, c_in, self._conf, self._inject_inputs)
            with torch.no_grad():
                gate_val = float(self.injectors[stage_idx].gamma.abs().mean().item())
                cos_sim = float(F.cosine_similarity(tokens, out, dim=-1).mean().item())
                self._stage_gate_vals[stage_idx] = gate_val
                self._stage_cos_sims[stage_idx] = cos_sim
                self._last_gate_val = sum(self._stage_gate_vals) / self._num_stages
                self._last_inject_cos_sim = sum(self._stage_cos_sims) / self._num_stages
                if stage_idx == 0:
                    # 注入殘差 = injector 輸出 − 輸入（Injector.forward 回傳 x + gamma*delta，
                    # 故 out-tokens 即實際注入的 gamma*delta）；比值化以對齊 sam_adapter。
                    _delta_norm = (out - tokens).norm(dim=-1).mean().item()
                    _vit_norm = tokens.norm(dim=-1).mean().item()
                    self._last_delta_norm_ratio = _delta_norm / (_vit_norm + 1e-8)
            return (out.reshape(B, H, W, C),)
        return hook

    def _make_extract_post_hook(self, stage_idx):
        def hook(module, inp, output):
            if self._c is None:
                return output                    # 未 set_features：不更新 c
            B, H, W, C = output.shape
            vit_tokens = output.reshape(B, H * W, C)
            first = self._extract_c[stage_idx] is None
            if first:
                self._extract_c[stage_idx] = self._c       # 首次觸發：快照輸入
            # 重放時必須以相同輸入重跑 extractor（checkpoint 依 op 順序回填
            # saved tensors），但只有首次觸發才推進 _c 狀態
            c_new = self.extractors[stage_idx](
                self._extract_c[stage_idx], vit_tokens,
                self._extract_inputs, self._scale_hw)
            if first:
                self._c = c_new
            return output  # 不改 ViT 輸出
        return hook
