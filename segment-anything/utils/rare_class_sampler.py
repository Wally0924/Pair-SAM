# segment-anything/utils/rare_class_sampler.py
"""
RareClassSampler — DAFormer 風格的稀有類別取樣（Hoyer et al., CVPR 2022）。

每抽一個樣本：
  1. 依 P(c) = softmax((1 - f(c)) / T) 抽一個類別 c（f(c) = 類別像素頻率）。
  2. 從「GT 含類別 c」的影像中均勻抽一張，回傳 dataset index。
無任何影像含之的類別，其 P(c) 設 0 並重正規化。所有隨機性由綁定 seed 的
torch.Generator 驅動（__init__ 建立、跨 __iter__ 沿用）→ 跨 run 可重現、跨 epoch 變化。
"""
from typing import List

import torch
from torch.utils.data import Sampler


class RareClassSampler(Sampler):
    def __init__(self, class_presence: List[List[int]], class_pixel_counts: List[int],
                 num_samples: int, temperature: float = 0.01, seed: int = 42,
                 num_classes: int = 19):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.temperature = temperature
        self.seed = seed

        counts = torch.tensor(class_pixel_counts, dtype=torch.float64)
        f = counts / counts.sum().clamp(min=1.0)
        probs = torch.softmax((1.0 - f) / temperature, dim=0)

        self.class_to_indices = {c: [] for c in range(num_classes)}
        for i, present in enumerate(class_presence):
            for c in present:
                if 0 <= c < num_classes:
                    self.class_to_indices[c].append(i)

        for c in range(num_classes):
            if len(self.class_to_indices[c]) == 0:
                probs[c] = 0.0
        probs = probs / probs.sum().clamp(min=1e-12)
        self.class_probs = probs.tolist()
        self._probs_t = probs

        self._g = torch.Generator()
        self._g.manual_seed(seed)

    def _draw_one_class(self) -> int:
        return int(torch.multinomial(self._probs_t, 1, generator=self._g).item())

    def _draw_one_index(self, c: int) -> int:
        idxs = self.class_to_indices[c]
        j = int(torch.randint(len(idxs), (1,), generator=self._g).item())
        return idxs[j]

    def __iter__(self):
        for _ in range(self.num_samples):
            c = self._draw_one_class()
            yield self._draw_one_index(c)

    def __len__(self):
        return self.num_samples
