import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedSemanticLoss(nn.Module):
    def __init__(self, ce_weight=1.0, focal_weight=2.0, dice_weight=2.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-5
        
        # 真正的 Focal Loss 超參數
        self.gamma = 2.0
        self.alpha = 0.25 # 平衡前景(1)與背景(0)的初始權重

        self.class_map = {
            "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
            "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
            "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
            "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
        }
        
        # 建立 Pixel-wise 的 CE Loss (忽略 255)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, fused_logits_hr, gt_mask, active_prompts):
        """
        Inputs:
            fused_logits_hr: (B, 19, H, W) 
            gt_mask: (B, H, W)
            active_prompts: list of strings (例如 ["road", "pole"])
        """
        device = fused_logits_hr.device
        metrics = {"total": 0.0, "ce": 0.0, "focal": 0.0, "dice": 0.0}
        
        # ==========================================
        # 1. 像素級別 CE Loss (全圖算)
        # ==========================================
        ce_loss = self.ce_loss_fn(fused_logits_hr, gt_mask)
        metrics["ce"] = ce_loss.item()
        
        # 如果使用者沒有要求 Focal 或 Dice，提早回傳加速
        if self.focal_weight <= 0 and self.dice_weight <= 0:
            total_loss = self.ce_weight * ce_loss
            metrics["total"] = total_loss.item()
            # 確保提早回傳時也有 ce, focal, dice 鍵值以免 Trainer 報錯
            metrics["ce"] = ce_loss.item()
            metrics["focal"] = 0.0
            metrics["dice"] = 0.0
            return total_loss, metrics

        # ==========================================
        # 2. 準備逐類別 (Class-wise) 計算的變數
        # ==========================================
        # 轉成機率佈局 (B, 19, H, W)
        pred_probs = F.softmax(fused_logits_hr, dim=1)
        
        # 製作過濾遮罩，排除 255 (B, H, W)
        valid_mask = (gt_mask != 255).float()
        
        total_focal = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)
        valid_prompts_count = 0

        # ==========================================
        # 3. 逐類別迴圈 (Non-vectorized!)
        # ==========================================
        for prompt in active_prompts:
            if prompt not in self.class_map:
                continue
                
            cls_id = self.class_map[prompt]
            valid_prompts_count += 1
            
            # (一) 抽出單一類別的預測機率 (B, H, W)
            p_t = pred_probs[:, cls_id, :, :]
            
            # 抽出該類別的「原始 Logit」(B, H, W)，專門給 BCE with logits 使用
            logit_t = fused_logits_hr[:, cls_id, :, :]
            
            # (二) 製作該類別專屬的 Ground Truth (B, H, W)，是這個類別的給 1，不是的給 0
            target_t = (gt_mask == cls_id).float()
            
            # (三) 套用 valid_mask，蓋掉 255 的邊緣背景區塊
            # 機率與目標都套上 valid_mask
            p_t_masked = p_t * valid_mask
            target_t_masked = target_t * valid_mask
            
            # --- 計算 Focal Loss ---
            p_t_clamped = torch.clamp(p_t_masked, min=1e-7, max=1.0-1e-7)
            prob_correct = p_t_clamped * target_t_masked + (1.0 - p_t_clamped) * (1.0 - target_t_masked)
            
            # 💡 基礎的分數：改用安全的 bce_with_logits！餵入原始的 logit_t
            bce_loss_raw = F.binary_cross_entropy_with_logits(logit_t, target_t_masked, reduction='none')
            # 蓋掉 255 的邊緣區域
            bce_loss = bce_loss_raw * valid_mask
            
            # Focal 調節因子
            focal_term = (1.0 - prob_correct) ** self.gamma
            
            # Alpha 調節因子
            alpha_term = self.alpha * target_t_masked + (1.0 - self.alpha) * (1.0 - target_t_masked)
            
            # 單一類別 Focal 總分 (平均到這張影像的合法像素上)
            focal_loss_pixel = alpha_term * focal_term * bce_loss
            focal_loss_cls = focal_loss_pixel.sum() / (valid_mask.sum() + 1e-6)
            
            # --- 計算 Dice Loss ---
            # 交集
            intersection = (p_t_clamped * target_t_masked).sum()
            # 聯集 (不扣掉交集，這是 Dice 特殊的聯集定義)
            union = p_t_clamped.sum() + target_t_masked.sum()
            
            # 計算 Dice
            dice_loss_cls = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            # 累加到總合中
            total_focal += focal_loss_cls
            total_dice += dice_loss_cls

        # ==========================================
        # 4. 平均與加權總結 (大總匯)
        # ==========================================
        if valid_prompts_count > 0:
            avg_focal = total_focal / valid_prompts_count
            avg_dice = total_dice / valid_prompts_count
        else:
            avg_focal = torch.tensor(0.0, device=device)
            avg_dice = torch.tensor(0.0, device=device)
            
        metrics["focal"] = avg_focal.item()
        metrics["dice"] = avg_dice.item()
        
        # 權重相承
        weighted_loss = (self.ce_weight * ce_loss) + (self.focal_weight * avg_focal) + (self.dice_weight * avg_dice)
        metrics["total"] = weighted_loss.item()
        
        # 🛡️ 防線
        if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
            print("⚠️ Warning: Combined Loss became NaN/Inf! Returning zero loss to prevent crash.")
            weighted_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return weighted_loss, metrics