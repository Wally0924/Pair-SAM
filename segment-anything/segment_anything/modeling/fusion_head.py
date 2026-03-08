import torch
import torch.nn as nn

class SemanticFusionHead(nn.Module):
    """
    A lightweight fusion head that takes N independent class logits 
    (e.g., from SAM's mask_decoder) and enforces mutual exclusivity 
    and spatial consistency across classes.
    """
    def __init__(self, num_classes: int = 19, hidden_dim: int = 64):
        super().__init__()
        # 1x1 Conv to mix class probabilities
        self.conv1 = nn.Conv2d(num_classes, hidden_dim, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # 3x3 Conv to smooth boundaries (Spatial Context)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        
        # Output back to num_classes
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input logits of shape (B, num_classes, H, W). 
               Typically these are the post-processed high-res masks.
        Returns:
            out: Fused logits of shape (B, num_classes, H, W)
        """
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.classifier(out)
        return out + identity
