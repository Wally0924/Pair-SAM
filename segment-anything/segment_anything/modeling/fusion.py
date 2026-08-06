import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from .common import LayerNorm2d
from .uawarpc_head import UAWarpCHead
from .cma_utils import warp, estimate_probability_of_confidence_interval


# =============================================================================
# VGG16AlignmentBackbone
#
# 為 UAWarpCHead 提供多尺度 VGG16 特徵，回傳 5 個尺度的特徵列表：
#   [stride2(64ch), stride4(128ch), stride8(256ch), stride16(512ch), stride32(512ch)]
# stride32（conv5）不參與 UAWarpCHead，供 DeformAdapter RPM 的真 1/32 尺度使用；
# CMA checkpoint 含 conv5 權重（features.24/26/28），load_cma_weights 一併載入。
#
# UAWarpCHead 使用 in_index=[2,3]，即取 stride8 與 stride16 兩個尺度：
#   - stride8  → 對 256×256 輸入為 32×32（UAWarpC level-3 constraint）
#   - stride16 → 對 256×256 輸入為 16×16（UAWarpC level-4 constraint）
# =============================================================================
class VGG16AlignmentBackbone(nn.Module):
    """Multi-scale VGG16 feature extractor matching CMA's alignment_backbone."""

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=None)
        feats = list(vgg.features)
        self.level0 = nn.Sequential(*feats[:5])    # stride 2,  64ch
        self.level1 = nn.Sequential(*feats[5:10])  # stride 4, 128ch
        self.level2 = nn.Sequential(*feats[10:17]) # stride 8, 256ch  ← in_index[0]=2
        self.level3 = nn.Sequential(*feats[17:24]) # stride 16, 512ch ← in_index[1]=3
        self.level4 = nn.Sequential(*feats[24:31]) # stride 32, 512ch（conv5，RPM 1/32）

    def forward(self, x: torch.Tensor):
        f0 = self.level0(x)
        f1 = self.level1(f0)
        f2 = self.level2(f1)
        f3 = self.level3(f2)
        f4 = self.level4(f3)
        return [f0, f1, f2, f3, f4]

    def load_cma_weights(self, vgg_state_dict: dict):
        """載入從 CMA checkpoint 抽取的 alignment_backbone 權重。

        CMA checkpoint 儲存的是完整 VGG module 的 state_dict，
        key 格式為 'features.0.weight'。
        torchvision VGG.features.state_dict() 的 key 格式為 '0.weight'（無前綴）。
        因此需要先剝除 'features.' 前綴才能正確比對。
        """
        # 剝除 'features.' 前綴，使 key 格式與 torchvision features.state_dict() 一致
        stripped = {
            (k[len('features.'):] if k.startswith('features.') else k): v
            for k, v in vgg_state_dict.items()
        }

        tmp_vgg = tvm.vgg16(weights=None)
        tmp_state = tmp_vgg.features.state_dict()
        matched = {k: v for k, v in stripped.items()
                   if k in tmp_state and v.shape == tmp_state[k].shape}
        print(f'[VGG16AlignmentBackbone] Loading {len(matched)}/{len(stripped)} VGG keys.')
        tmp_vgg.features.load_state_dict(matched, strict=False)

        feats = list(tmp_vgg.features)
        self.level0.load_state_dict(nn.Sequential(*feats[:5]).state_dict(), strict=False)
        self.level1.load_state_dict(nn.Sequential(*feats[5:10]).state_dict(), strict=False)
        self.level2.load_state_dict(nn.Sequential(*feats[10:17]).state_dict(), strict=False)
        self.level3.load_state_dict(nn.Sequential(*feats[17:24]).state_dict(), strict=False)
        self.level4.load_state_dict(nn.Sequential(*feats[24:31]).state_dict(), strict=False)
        print('[VGG16AlignmentBackbone] ✅ Loaded CMA alignment_backbone weights.')


# =============================================================================
# CMAAlignment
#
# 以 CMA 的 UAWarpC 稠密匹配執行跨條件影像特徵對齊。
#
# 設計：
#   1. VGG16AlignmentBackbone（凍結）：從原始影像提取幾何特徵
#   2. UAWarpCHead（凍結）：計算 flow field + uncertainty map
#   3. warp()：將 f_ref（ViT-H embedding）扭曲到 f_curr 視角
#   4. confidence：exp(-uncertainty) × boundary validity mask
# =============================================================================
class CMAAlignment(nn.Module):
    """
    UAWarpC-based dense feature alignment (CMA, Bruggemann et al., ICCV 2023).

    VGG16 backbone 與 UAWarpCHead 在載入 CMA pretrained 權重後皆凍結；
    本模組本身不含任何可訓練參數，僅輸出對齊後的 reference 特徵與 confidence map，
    供下游 PairSAM.vgg_injector (MultiScaleCrossAttnInjector) 使用。
    """

    _IMG_MEAN = [0.485, 0.456, 0.406]
    _IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        embed_dim: int = 256,
        pretrained_path: str = None,
        confidence_threshold: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conf_threshold = confidence_threshold

        self.vgg_backbone = VGG16AlignmentBackbone()
        for p in self.vgg_backbone.parameters():
            p.requires_grad_(False)

        # in_index=[2,3] 選取 stride8（256ch）與 stride16（512ch）兩個尺度
        self.warp_head = UAWarpCHead(
            in_index=[2, 3],
            input_transform='multiple_select',
            estimate_uncertainty=True,
        )
        for p in self.warp_head.parameters():
            p.requires_grad_(False)

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self._last_conf_mean: float = 0.0
        self._last_valid_ratio: float = 0.0
        self._last_flow = None
        self._last_confidence_map = None

    def _load_pretrained(self, path: str):
        weights = torch.load(path, map_location='cpu')
        print(f'[CMAAlignment] checkpoint keys: {list(weights.keys())}')
        if 'vgg' in weights:
            self.vgg_backbone.load_cma_weights(weights['vgg'])
        if 'uawarpc' in weights:
            missing, unexpected = self.warp_head.load_state_dict(
                weights['uawarpc'], strict=False)
            print(f'[UAWarpCHead] Missing: {len(missing)}, Unexpected: {len(unexpected)}')
            if missing:
                print(f'[UAWarpCHead] Missing keys (first 5): {missing[:5]}')
            if len(missing) == 0:
                print('[CMAAlignment] ✅ Loaded UAWarpCHead weights from CMA checkpoint.')
            else:
                print('[CMAAlignment] ⚠️  UAWarpCHead partially loaded — check missing keys above.')
        else:
            print('[CMAAlignment] ⚠️  No "uawarpc" key in checkpoint — UAWarpCHead uses random init!')

    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """ImageNet 正規化，img 值域假設為 [0, 255]。"""
        mean = torch.tensor(self._IMG_MEAN, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(self._IMG_STD,  device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        return (img / 255.0 - mean) / std

    @torch.no_grad()
    def _extract_vgg_features(self, img: torch.Tensor):
        """
        從原始影像提取兩組 VGG 特徵供 UAWarpCHead 使用：
          - feats     : 完整解析度（1024×1024），用於 level-1/2
          - feats_256 : 縮放至 256×256，level-3/4 需要 32×32 和 16×16
        """
        img_norm = self._normalize_image(img)
        feats = self.vgg_backbone(img_norm)

        img_256 = F.interpolate(img_norm, size=(256, 256), mode='bilinear', align_corners=False)
        feats_256 = self.vgg_backbone(img_256)
        return feats, feats_256

    @torch.no_grad()
    def pre_align(
        self,
        img_curr: torch.Tensor,          # (B, 3, H, W) adverse weather image, values in [0, 255]
        img_ref: torch.Tensor,           # (B, 3, H, W) clear reference image, values in [0, 255]
        out_size: tuple = (64, 64),      # target feature map spatial size
        l2_native: bool = False,         # if True, return l2 at 2×out_size (genuine stride-8)
    ) -> dict:
        """
        Run VGG feature extraction + UAWarpC alignment BEFORE the ViT encoder.

        UAWarpC operates purely on VGG features of raw images, so there is no
        circular dependency with ViT embeddings. This method can be called once
        per forward pass before the SAM image encoder runs.

        Returns:
            dict with keys:
                'l2': (B, 256, out_H, out_W)
                    Warped VGG level-2 (256ch, stride-8) features from the clear
                    reference image, aligned to the adverse-weather image viewpoint.
                'l3': (B, 512, out_H, out_W)
                    Warped VGG level-3 (512ch, stride-16) features from the clear
                    reference image, aligned to the adverse-weather image viewpoint.
                'l4': (B, 512, out_H//2, out_W//2)
                    Warped VGG level-4 (conv5, 512ch, genuine stride-32) features,
                    warped with the same flow downsampled ×0.5 (pixel-unit scaling).
            Features are returned un-zeroed; the continuous confidence map (with
            boundary validity baked in) is returned as 'mask' for soft, CMA-style
            weighting downstream in the Injector (feat × conf).
        """
        out_H, out_W = out_size

        # Step 1: Extract VGG features from both images
        feats_curr, feats_curr_256 = self._extract_vgg_features(img_curr)
        feats_ref,  feats_ref_256  = self._extract_vgg_features(img_ref)

        # Step 2: Run UAWarpCHead → get flow and uncertainty at finest level (index 3)
        results = self.warp_head(
            trg=feats_curr,
            src=feats_ref,
            trg_256=feats_curr_256,
            src_256=feats_ref_256,
            out_size=out_size,
        )
        flow1, uncertainty1 = results[3]

        # Step 3: Resize flow / uncertainty to out_size if needed
        if flow1.shape[-2:] != (out_H, out_W):
            flow1 = F.interpolate(flow1, size=(out_H, out_W), mode='bilinear', align_corners=False)
            uncertainty1 = F.interpolate(uncertainty1, size=(out_H, out_W), mode='bilinear', align_corners=False)

        # Step 4: Resize VGG level-3 (index 3 = stride-16 = 512ch) ref features to out_size
        f_vgg_ref_l3 = feats_ref[3]  # (B, 512, H_vgg, W_vgg)
        if f_vgg_ref_l3.shape[-2:] != (out_H, out_W):
            f_vgg_ref_l3 = F.interpolate(f_vgg_ref_l3, size=(out_H, out_W),
                                          mode='bilinear', align_corners=False)

        # Step 4b: Resize VGG level-2 (index 2 = stride-8 = 256ch) ref features to out_size
        f_vgg_ref_l2 = feats_ref[2]  # (B, 256, H_vgg, W_vgg) — stride8, 2× spatial res
        if f_vgg_ref_l2.shape[-2:] != (out_H, out_W):
            f_vgg_ref_l2 = F.interpolate(f_vgg_ref_l2, size=(out_H, out_W),
                                          mode='bilinear', align_corners=False)

        # Step 5: Warp both VGG scales to adverse viewpoint using the same flow field
        f_ref_warped_l3, validity_mask = warp(f_vgg_ref_l3, flow1, return_mask=True)
        f_ref_warped_l2, _ = warp(f_vgg_ref_l2, flow1, return_mask=True)

        # Step 6: Confidence = UAWarpC uncertainty → probability × boundary validity
        confidence = estimate_probability_of_confidence_interval(uncertainty1)  # (B, 1, out_H, out_W)
        confidence = confidence * validity_mask.unsqueeze(1).float()

        # Step 7: CMA-style soft gating（取代逐像素硬歸零）。特徵原樣輸出，連續
        # confidence（Step 6 已乘 validity，無對應像素為 0）作為 'mask' 交給 RPM→Injector，
        # 由 injector 端 feat×conf 做單一軟加權（等同 CMA Sec 3.3 confidence-weighted pooling）。

        # Step 7b: VGG level-4（conv5，真 stride-32）→ 半解析度 warp。
        # flow 是 pixel-unit：grid 縮小一半，位移數值也要 ×0.5（與 l2_native ×2 同一套邏輯）。
        l4H, l4W = out_H // 2, out_W // 2
        f_vgg_ref_l4 = feats_ref[4]  # (B, 512, H/32, W/32)
        if f_vgg_ref_l4.shape[-2:] != (l4H, l4W):
            f_vgg_ref_l4 = F.interpolate(f_vgg_ref_l4, size=(l4H, l4W),
                                          mode='bilinear', align_corners=False)
        flow_l4 = F.interpolate(
            flow1, size=(l4H, l4W), mode='bilinear', align_corners=False) * 0.5
        f_ref_warped_l4, _ = warp(f_vgg_ref_l4, flow_l4, return_mask=True)  # 原樣，軟加權在 injector

        # Step 8: Update diagnostic attributes
        self._last_conf_mean      = float(confidence.mean().item())
        self._last_valid_ratio    = float(validity_mask.float().mean().item())
        self._last_flow           = flow1.cpu()           # (B, 2, out_H, out_W)
        self._last_confidence_map = confidence.cpu()      # (B, 1, out_H, out_W)

        # Step 9 (opt-in): native stride-8 l2 at 2×out_size
        if l2_native:
            l2H, l2W = out_H * 2, out_W * 2
            f_l2 = feats_ref[2]                                # native stride-8 (256ch)
            if f_l2.shape[-2:] != (l2H, l2W):
                f_l2 = F.interpolate(f_l2, size=(l2H, l2W), mode='bilinear', align_corners=False)
            # CRITICAL: warp() uses pixel-unit flow (vgrid = pixel_grid + flo).
            # Upsampling the flow from out_size to 2×out_size doubles the grid resolution,
            # so displacement VALUES must be scaled ×2 to preserve the same physical shift.
            # Omitting ×2 silently halves the alignment displacement → misaligned reference.
            flow_l2 = F.interpolate(
                flow1, size=(l2H, l2W), mode='bilinear', align_corners=False) * 2.0
            f_l2_warped, _ = warp(f_l2, flow_l2, return_mask=True)
            # 連續 confidence 上採至最細尺度（2×），供 RPM 逐尺度 pool 下去做軟加權
            conf_l2 = F.interpolate(confidence, size=(l2H, l2W),
                                    mode='bilinear', align_corners=False)
            return {
                'l2':   f_l2_warped,           # (B, 256, 2×out_H, 2×out_W) genuine stride-8
                'l3':   f_ref_warped_l3,       # (B, 512, out_H, out_W)
                'l4':   f_ref_warped_l4,       # (B, 512, out_H/2, out_W/2) genuine stride-32
                'mask': conf_l2,               # (B, 1, 2×out_H, 2×out_W) 連續信心，RPM pools down
            }

        return {
            'l2':   f_ref_warped_l2,          # (B, 256, out_H, out_W)
            'l3':   f_ref_warped_l3,          # (B, 512, out_H, out_W)
            'l4':   f_ref_warped_l4,          # (B, 512, out_H/2, out_W/2) genuine stride-32
            'mask': confidence,               # (B, 1,   out_H, out_W) 連續信心 [0,1]（軟加權）
        }

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


# =============================================================================
# FlowGuidedSemanticAlignment
#
# 設計：幾何對齊（CMAAlignment）→ Reference Conditioning Module（本模組）
#
# Step 1 (CMAAlignment 完成)：VGG flow → warp(f_ref) → f_ref_warped + confidence
#
# Step 2 (本模組)：Multi-channel delta conditioning
#   delta = Conv2layer(concat(f_curr, f_ref_warped))   # 256-channel，每維度獨立學習補償量
#   delta = delta * conf_map                            # confidence soft gate 作用在輸出端
#   conditioned = f_curr + delta                        # residual，f_curr 語意完整保留
#
# 與舊版 scalar alpha 的差異：
#   舊版：alpha[1ch] * f_ref_warped → 所有 256 channel 被同一個值縮放
#   新版：delta[256ch] → 每個 channel 獨立決定修正方向與幅度
# =============================================================================
class FlowGuidedSemanticAlignment(nn.Module):
    """
    Reference Conditioning Module for ViT embeddings.

    Computes a 256-channel residual delta from (f_curr, f_ref_warped),
    soft-gates it with the UAWarpC confidence map, and adds it to f_curr.
    """

    def __init__(self, embed_dim: int = 256, confidence_threshold: float = 0.2):
        super().__init__()
        self.conf_threshold = confidence_threshold

        # 2-layer 1×1 Conv：學習 f_curr 與 f_ref_warped 的跨域關係 → 256-ch delta
        # 初始化接近零輸出，訓練初期不干擾 f_curr
        self.delta_net = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        nn.init.zeros_(self.delta_net[2].weight)
        nn.init.zeros_(self.delta_net[2].bias)

        self.norm = LayerNorm2d(embed_dim)

    def forward(
        self,
        f_curr:       torch.Tensor,  # [B, C, H, W]  ViT embedding（惡劣天氣）
        f_ref_warped: torch.Tensor,  # [B, C, H, W]  幾何對齊後的 f_ref（CMAAlignment 輸出）
        confidence:   torch.Tensor,  # [B, 1, H, W]  UAWarpC confidence map（來自 CMAAlignment）
    ) -> torch.Tensor:
        # 256-channel delta，表達能力遠高於 scalar alpha
        delta = self.delta_net(torch.cat([f_curr, f_ref_warped], dim=1))

        # Soft confidence gate 作用在輸出端：梯度可流過低信心區，但貢獻被壓低
        # 比 hard mask 更好：grad 不斷，低信心區的 delta 學習趨近於零即可
        delta = delta * confidence.clamp(0.0, 1.0)

        # Residual injection：f_curr 語意完整保留，delta 只做補償
        f_fuse = f_curr + delta

        with torch.no_grad():
            self._last_delta_norm = float(delta.detach().norm(dim=1).mean().item())
            self._last_conf_mean  = float(confidence.mean().item())
            cos_sim = F.cosine_similarity(f_curr, f_ref_warped, dim=1, eps=1e-8)
            self._last_cos_sim    = cos_sim.detach().cpu()

        return self.norm(f_fuse)


# 保留供消融實驗對比（無 cross-attention 的純幾何 gate 版本）
class ConfidenceGatedFusion(nn.Module):
    """Ablation baseline: geometric gate only, no semantic cross-attention."""

    def __init__(self, embed_dim: int = 256, confidence_threshold: float = 0.2):
        super().__init__()
        self.conf_threshold = confidence_threshold
        self.gate_net = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, f_curr, f_aligned, confidence):
        conf_patch = F.adaptive_avg_pool2d(confidence, 7)
        conf_hard  = F.interpolate((conf_patch >= self.conf_threshold).float(),
                                   size=f_curr.shape[-2:], mode='nearest')
        conf_soft  = F.interpolate(conf_patch.clamp(0.0, 1.0),
                                   size=f_curr.shape[-2:], mode='bilinear', align_corners=False)
        alpha = self.gate_net(torch.cat([f_curr, f_aligned], dim=1))
        alpha = alpha * conf_hard * conf_soft
        return self.norm((1.0 - alpha) * f_curr + alpha * f_aligned)
