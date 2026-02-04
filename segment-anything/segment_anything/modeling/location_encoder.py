import torch
import torch.nn as nn
import numpy as np

# 假設 layers.py 和 functional.py 放在名為 rff 的資料夾中
# 如果放在同級目錄，請改成 from layers import GaussianEncoding
from .rff.layers import GaussianEncoding

# ==========================================
# 1. 投影函式: Equal Earth Projection (維持 GeoCLIP 原作)
# ==========================================
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L):
    """
    將經緯度 (Lat, Lon) 投影到平面座標 (x, y)
    L: (Batch, 2) -> [Latitude, Longitude]
    """
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    sin_theta = torch.clamp(sin_theta, -1.0, 1.0) # 數值穩定性保護
    
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    
    return (torch.stack((x, y), dim=1) * SF) / 180

# ==========================================
# 2. 核心模組: Location Encoder
# ==========================================
class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma, embed_dim=256):
        super(LocationEncoderCapsule, self).__init__()
        
        # 使用 rff 庫中的 GaussianEncoding
        # Input: 2 (x, y)
        # encoded_size=256 -> 內部會產生 256 個頻率
        # Output: sin + cos -> 256 * 2 = 512 維
        self.rff_encoding = GaussianEncoding(
            sigma=sigma, 
            input_size=2, 
            encoded_size=256
        )
        
        # MLP 架構 (GeoCLIP 原始設計)
        # 輸入必須是 512 (因為 encoded_size=256, sin/cos 擴展後變 512)
        self.capsule = nn.Sequential(
            self.rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # 輸出層: 映射到 SAM 的維度 (通常是 256)
        self.head = nn.Sequential(nn.Linear(1024, embed_dim))

    def forward(self, x):
        # x: (Batch, 2)
        feat = self.capsule(x)
        out = self.head(feat)
        return out

class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], embed_dim=256):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.embed_dim = embed_dim

        # 建立多尺度編碼器 (Multi-scale Encoders)
        # 透過 ModuleList 註冊子模組
        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s, embed_dim=embed_dim))

    def forward(self, location):
        """
        Args:
            location: (Batch, 2) Tensor, containing [Lat, Lon]
        Returns:
            location_features: (Batch, embed_dim)
        """
        # 1. 投影: (Batch, 2)
        location_proj = equal_earth_projection(location)
        
        # 2. 初始化輸出容器
        location_features = torch.zeros(location_proj.shape[0], self.embed_dim).to(location.device)

        # 3. 加總不同尺度的特徵 (Multi-scale Summation)
        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location_proj)
        
        return location_features