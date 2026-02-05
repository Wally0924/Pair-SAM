import torch
import torch.nn as nn
import numpy as np
import math

# ==========================================
# 1. Functional & Layers (源自 rff/functional.py & layers.py)
# ==========================================
class GaussianEncoding(nn.Module):
    """
    對應原 layers.GaussianEncoding 與 functional.gaussian_encoding
    """
    def __init__(self, sigma, input_size=2, encoded_size=256):
        super().__init__()
        self.sigma = sigma
        self.input_size = input_size
        self.encoded_size = encoded_size
        
        # 建立隨機高斯矩陣 B (固定不更新)
        # shape: (encoded_size, input_size) -> (256, 2)
        b = torch.randn((encoded_size, input_size)) * sigma
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v):
        # v: (Batch, 2)
        # proj: (Batch, 256)
        vp = 2 * np.pi * (v @ self.b.T)
        # output: (Batch, 512) -> [cos, sin]
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

# ==========================================
# 2. Equal Earth Projection (源自 location_encoder.py)
# ==========================================
# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L):
    """
    將經緯度投影到平面座標
    L: (Batch, 2) -> [Latitude, Longitude]
    """
    latitude = L[:, 0]
    longitude = L[:, 1]
    
    # 轉弧度
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    
    sin_theta = (math.sqrt(3) / 2) * torch.sin(latitude_rad)
    # 數值穩定性 clamp
    sin_theta = torch.clamp(sin_theta, -1.0, 1.0) 
    theta = torch.asin(sin_theta)
    
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    
    x = (2 * math.sqrt(3) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    
    return (torch.stack((x, y), dim=1) * SF) / 180

# ==========================================
# 3. Location Encoder Capsules (源自 location_encoder.py)
# ==========================================
class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        # RFF Encoding: Input 2 -> Output 512 (256*2)
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        
        # GeoCLIP 原始 MLP 結構
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        # 原始 Head 輸出 512
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

# ==========================================
# 4. Main Location Encoder (整合適配)
# ==========================================
class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], output_dim=256, pretrained_path=None):
        """
        Args:
            sigma: RFF 的頻率參數 (GeoCLIP 預設三層)
            output_dim: 目標輸出維度 (SAM 為 256)
            pretrained_path: 若有 GeoCLIP 預訓練權重檔路徑則填入
        """
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        
        # 建立 Capsules (保持原始結構以利載入權重)
        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        # 載入權重 (如果有的話)
        if pretrained_path:
            print(f"Loading Location Encoder weights from {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                # 過濾掉不匹配的鍵值 (如 projection layer)
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)
            except Exception as e:
                print(f"Warning: Failed to load weights: {e}")

        # [新增] 投影層: 512 -> output_dim (256)
        # 由於 GeoCLIP 原始輸出是 512，我們需要這一層來對齊 SAM
        self.output_projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, location):
        """
        Args:
            location: (Batch, 2) tensor, [Lat, Lon]
        Returns:
            location_features: (Batch, output_dim) usually 256
        """
        # 1. 投影
        location = equal_earth_projection(location)
        
        # 2. 多尺度特徵提取與加總
        location_features = torch.zeros(location.shape[0], 512).to(location.device)
        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        
        # 3. 投影到 SAM 維度 (512 -> 256)
        out = self.output_projection(location_features)
        
        return out