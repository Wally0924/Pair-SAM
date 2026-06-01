# segment-anything/segment_anything/modeling/semantic_assembly.py
"""
語意 logits 組裝共用函式。

歷史背景：原本「將 K 個 active class 的 low-res logits scatter 進 19 類別張量
（缺席類別填 -10.0），再選擇性套用 context_fusion_head (LRH)」這段邏輯散落在
trainer(train/validate)、eval、viz、inference 共 7 處。為了讓 --lrh 開關成為單一
真值來源、並杜絕 train/eval 套用不一致，集中於此。
"""
from typing import List, Optional

import torch
import torch.nn as nn


def assemble_semantic_logits(
    low_res_logits: torch.Tensor,      # (K, H, W) — 每個 active class 一張 logit map
    class_ids: List[int],              # len=K，對應 0..num_classes-1
    fusion_head: Optional[nn.Module],  # ResidualDWConvFusion；use_lrh=False 時可為 None
    *,
    num_classes: int = 19,
    use_lrh: bool = True,
    fill_value: float = -10.0,
) -> torch.Tensor:
    """組裝 (1, num_classes, H, W) 語意 logits，並依 use_lrh 決定是否套用 LRH。

    Returns:
        (1, num_classes, H, W) logits（use_lrh=True 時為 LRH 精修後）。
    """
    K, H, W = low_res_logits.shape
    full = torch.full(
        (1, num_classes, H, W), fill_value,
        device=low_res_logits.device, dtype=low_res_logits.dtype,
    )
    for k, c in enumerate(class_ids):
        full[0, c] = low_res_logits[k]

    if use_lrh:
        if fusion_head is None:
            raise ValueError("use_lrh=True 但 fusion_head 為 None")
        return fusion_head(full)
    return full
