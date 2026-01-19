import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, focal_weight=20.0, dice_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

        # 定義類別映射表 (Prompt文字 -> GT中的ID)
        # ⚠️ 注意：必須與您 dataloader.py 中的 CLASS_MAP 完全一致
        self.class_map = {
            "road": 0,
            "sidewalk": 1,
            "building": 2,
            "wall": 3,
            "fence": 4,
            "pole": 5,
            "traffic light": 6,
            "traffic sign": 7,
            "vegetation": 8,
            "terrain": 9,
            "sky": 10,
            "person": 11,
            "rider": 12,
            "car": 13,
            "truck": 14,
            "bus": 15,
            "train": 16,
            "motorcycle": 17,
            "bicycle": 18
        }

    def forward(self, pred_masks, gt_mask, iou_predictions, text_prompts):
        """
        計算單張影像(包含多個 Prompt)的 Loss
        
        Args:
            pred_masks (Tensor): (K, 3, H, W) 模型預測的 Logits (Float)，K 為 Prompt 數量。
            gt_mask (Tensor): (H, W) 真實標籤 ID Map (Int64/Long)。
            iou_predictions (Tensor): (K, 3) 模型預測的 IoU 信心分數。
            text_prompts (List[str]): 長度為 K 的文字提示列表。

        Returns:
            total_loss: 純量 Tensor
            metrics: 包含各項 loss數值的字典
        """
        device = pred_masks.device
        
        # ------------------------------------------------------------------
        # 1. 準備 Ground Truth (ID Map -> Binary Masks)
        # ------------------------------------------------------------------
        target_masks = []
        valid_indices = [] # 記錄哪些 prompt 是有效的 (有對應到 class_map)

        for i, prompt in enumerate(text_prompts):
            if prompt in self.class_map:
                class_id = self.class_map[prompt]
                # 製作 Binary Mask: 只有該類別的位置是 1.0
                binary_gt = (gt_mask == class_id).float()
                target_masks.append(binary_gt)
                valid_indices.append(i)
        
        # 防呆：如果這一批 prompts 裡沒有任何一個在 class_map 中 (極少見)
        if not target_masks:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {"total": 0.0, "bce": 0.0, "dice": 0.0, "iou_mse": 0.0}

        # 堆疊 Targets: (K_valid, H, W) -> (K_valid, 1, H, W)
        targets = torch.stack(target_masks).unsqueeze(1).to(device)

        # ------------------------------------------------------------------
        # 2. 選擇最佳預測 (Selection Strategy)
        # ------------------------------------------------------------------
        # 根據 iou_predictions 選出最有信心的那個 mask
        # 篩選出有效的預測
        valid_preds = pred_masks[valid_indices]      # (K_valid, 3, H, W)
        valid_iou_pred = iou_predictions[valid_indices] # (K_valid, 3)

        # 找出每個 prompt 分數最高的 index (0, 1, or 2)
        best_mask_indices = torch.argmax(valid_iou_pred, dim=1) # (K_valid,)

        # Gather: 取出最佳 Mask 與對應的 IoU 分數
        # 我們希望從 (K, 3, H, W) 變成 (K, 1, H, W)
        final_preds = []
        final_iou_conf = []

        for k, best_idx in enumerate(best_mask_indices):
            final_preds.append(valid_preds[k, best_idx, :, :])
            final_iou_conf.append(valid_iou_pred[k, best_idx])
        
        # 堆疊回 Tensor
        final_preds = torch.stack(final_preds).unsqueeze(1) # (K_valid, 1, H, W)
        final_iou_conf = torch.stack(final_iou_conf).unsqueeze(1) # (K_valid, 1)

        # ------------------------------------------------------------------
        # 3. 計算 Loss
        # ------------------------------------------------------------------
        
        # A. Dice Loss
        pred_prob = torch.sigmoid(final_preds)
        intersection = (pred_prob * targets).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        # B. Focal Loss (使用 BCEWithLogits 近似)
        # 這裡直接用 BCE，配合外層的 focal_weight 加權
        bce_loss = F.binary_cross_entropy_with_logits(final_preds, targets)
        
        # C. IoU Prediction Loss (MSE)
        # 計算"真實"的 IoU
        with torch.no_grad():
            pred_binary = (pred_prob > 0.5).float()
            inter = (pred_binary * targets).sum(dim=(2, 3))
            uni = pred_binary.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
            actual_iou = inter / (uni + 1e-5) # (K_valid, 1)

        iou_loss = F.mse_loss(final_iou_conf, actual_iou)
        
        # D. 總 Loss
        total_loss = (self.focal_weight * bce_loss) + \
                     (self.dice_weight * dice_loss) + \
                     (self.iou_weight * iou_loss)
        
        return total_loss, {
            "total": total_loss.item(),
            "bce": bce_loss.item(),
            "dice": dice_loss.item(),
            "iou_mse": iou_loss.item()
        }