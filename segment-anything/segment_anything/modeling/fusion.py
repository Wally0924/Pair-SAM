import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from .common import LayerNorm2d
from .uawarpc_head import UAWarpCHead
from .cma_utils import warp, estimate_probability_of_confidence_interval

try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from utils.viz_warp import save_warp_comparison
    _VIZ_AVAILABLE = True
except Exception:
    _VIZ_AVAILABLE = False


# =============================================================================
# VGG16AlignmentBackbone
#
# 為 UAWarpCHead 提供多尺度 VGG16 特徵，回傳 4 個尺度的特徵列表：
#   [stride2(64ch), stride4(128ch), stride8(256ch), stride16(512ch)]
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

    def forward(self, x: torch.Tensor):
        f0 = self.level0(x)
        f1 = self.level1(f0)
        f2 = self.level2(f1)
        f3 = self.level3(f2)
        return [f0, f1, f2, f3]

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

    VGG16 backbone and UAWarpCHead are both frozen after loading CMA pretrained
    weights. Only ConfidenceGatedFusion's gate_net is trainable.
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
        conf_threshold: float = 0.2,     # hard mask threshold (matches CMA paper Sec 3.3)
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
            Positions where confidence < conf_threshold are zeroed out (hard mask).
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

        # Step 7: Hard mask — matches CMA paper Sec 3.3 (conf < 0.2 → zero out)
        hard_mask = (confidence >= conf_threshold).float()  # (B, 1, out_H, out_W)
        f_ref_warped_l3_masked = f_ref_warped_l3 * hard_mask  # (B, 512, out_H, out_W)
        f_ref_warped_l2_masked = f_ref_warped_l2 * hard_mask  # (B, 256, out_H, out_W)

        # Step 8: Update diagnostic attributes
        self._last_conf_mean      = float(confidence.mean().item())
        self._last_valid_ratio    = float(validity_mask.float().mean().item())
        self._last_flow           = flow1.cpu()           # (B, 2, out_H, out_W)
        self._last_confidence_map = confidence.cpu()      # (B, 1, out_H, out_W)

        return {
            'l2':   f_ref_warped_l2_masked,   # (B, 256, out_H, out_W)
            'l3':   f_ref_warped_l3_masked,   # (B, 512, out_H, out_W)
            'mask': hard_mask,                 # (B, 1,   out_H, out_W) — 1=valid, 0=ignore
        }


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
