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

    def forward(
        self,
        f_curr:   torch.Tensor,        # [B, 256, 64, 64]  ViT-H embedding（惡劣天氣）
        f_ref:    torch.Tensor,        # [B, 256, 64, 64]  ViT-H embedding（晴天參考）
        img_curr: torch.Tensor = None, # [B, 3, 1024, 1024] 原始影像（值域 0-255）
        img_ref:  torch.Tensor = None, # [B, 3, 1024, 1024] 原始影像（值域 0-255）
    ):
        _, _, H, W = f_curr.shape

        if img_curr is not None and img_ref is not None:
            feats_curr, feats_curr_256 = self._extract_vgg_features(img_curr)
            feats_ref,  feats_ref_256  = self._extract_vgg_features(img_ref)
        else:
            # Fallback：ViT 特徵複製為 4 個尺度（精度較低但不崩潰）
            feats_curr = feats_curr_256 = [f_curr] * 4
            feats_ref  = feats_ref_256  = [f_ref]  * 4

        # 取最細的 level（index 3）flow 與 uncertainty
        results = self.warp_head(
            trg=feats_curr,
            src=feats_ref,
            trg_256=feats_curr_256,
            src_256=feats_ref_256,
            out_size=(H, W),
        )
        flow1, uncertainty1 = results[3]

        if flow1.shape[-2:] != (H, W):
            flow1 = F.interpolate(flow1, size=(H, W), mode='bilinear', align_corners=False)
            uncertainty1 = F.interpolate(uncertainty1, size=(H, W), mode='bilinear', align_corners=False)

        f_ref_warped, validity_mask = warp(f_ref, flow1, return_mask=True)

        confidence = estimate_probability_of_confidence_interval(uncertainty1)  # [B,1,H,W]
        confidence = confidence * validity_mask.unsqueeze(1).float()

        # Step 2 驗證：warp 前後 cos_sim 對比（每 200 次 forward 印一次，避免 log 爆炸）
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        if self._fwd_count % 200 == 1:
            with torch.no_grad():
                _cos_before = F.cosine_similarity(f_curr, f_ref, dim=1).mean().item()
                _cos_after  = F.cosine_similarity(f_curr, f_ref_warped, dim=1).mean().item()
                _flow_mag   = (flow1[:, 0]**2 + flow1[:, 1]**2).sqrt().mean().item()
                print(f'[CMAAlignment #{self._fwd_count}] '
                      f'cos_before={_cos_before:.4f}  cos_after={_cos_after:.4f}  '
                      f'delta={_cos_after - _cos_before:+.4f}  '
                      f'flow_mag={_flow_mag:.3f}feat_px  '
                      f'conf_mean={confidence.mean().item():.4f}')

                # 可視化：存 warp 對比圖（需要有原始影像輸入）
                if _VIZ_AVAILABLE and img_curr is not None and img_ref is not None:
                    save_warp_comparison(
                        img_curr=img_curr,
                        img_ref=img_ref,
                        flow=flow1,
                        confidence=confidence,
                        step=self._fwd_count,
                        out_dir='debug_viz/warp',
                    )

        # Hard confidence threshold: discard positions where confidence < threshold
        # confidence only serves as a binary spatial filter (keep/discard), nothing else
        hard_mask = (confidence >= self.conf_threshold).float()  # [B, 1, H, W]
        f_ref_warped = f_ref_warped * hard_mask

        with torch.no_grad():
            self._last_conf_mean      = float(confidence.mean().item())
            self._last_valid_ratio    = float(validity_mask.float().mean().item())
            # Stored for alignment visualization (CPU tensors; detached to avoid holding grad graph)
            self._last_flow           = flow1.detach().cpu()           # (B, 2, H, W) pixel displacement
            self._last_confidence_map = confidence.detach().cpu()      # (B, 1, H, W)

        return f_ref_warped, confidence


# =============================================================================
# FlowGuidedSemanticAlignment
#
# 設計：幾何對齊（CMAAlignment）→ Confidence Gate（本模組）
#
# Step 1 (CMAAlignment 完成)：VGG flow → warp(f_ref) → f_ref_warped
#   幾何粗對齊：解決相機視角差異、場景位移等大範圍幾何偏差
#
# Step 2 (本模組)：Feature-content Gate
#   gate_net 接收 concat(f_curr, f_ref_warped)，學習每個像素注入多少晴天特徵。
#   低信心區（動態物件、遮蔽）→ alpha→0，保留 f_curr 不受噪聲污染。
# =============================================================================
class FlowGuidedSemanticAlignment(nn.Module):
    """
    Geometric-then-gate alignment for ViT embeddings.

    CMAAlignment (upstream) handles geometric warp + confidence pooling.
    This module gates how much of the warped clear-weather feature is
    injected into f_curr via a learned per-pixel alpha map.
    """

    def __init__(self, embed_dim: int = 256, confidence_threshold: float = 0.2):
        super().__init__()
        self.conf_threshold = confidence_threshold

        # Gate: concat(f_curr, f_ref_warped) → per-pixel blend ratio α ∈ [0,1]
        self.gate_net = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(
        self,
        f_curr:       torch.Tensor,  # [B, C, H, W]  ViT embedding（惡劣天氣）
        f_ref_warped: torch.Tensor,  # [B, C, H, W]  幾何對齊後的 f_ref（CMAAlignment 輸出）
    ) -> torch.Tensor:
        # Gate: learn per-pixel injection ratio from feature content
        alpha = self.gate_net(torch.cat([f_curr, f_ref_warped], dim=1))

        f_fuse = f_curr + alpha * f_ref_warped  # additive injection; f_curr fully preserved

        with torch.no_grad():
            self._last_alpha   = alpha.detach().cpu()                         # [B, 1, H, W]
            cos_sim = F.cosine_similarity(f_curr, f_ref_warped, dim=1, eps=1e-8)
            self._last_cos_sim = cos_sim.detach().cpu()                       # [B, H, W]

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
