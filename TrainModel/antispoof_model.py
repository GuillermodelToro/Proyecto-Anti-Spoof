# antispoof_model
import re
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

IMG_SIZE = 320
META_DIM = 5  # [x_rel, y_rel, w_rel, h_rel, conf]

def default_norm():
    w = EfficientNet_V2_M_Weights.DEFAULT.transforms()
    return transforms.Normalize(mean=w.mean, std=w.std)

def build_img_transform(size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        default_norm()
    ])

def parse_meta_line(text: str) -> np.ndarray:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.strip())
    v = [float(x) for x in nums[:META_DIM]]
    if len(v) < META_DIM:
        v += [0.0] * (META_DIM - len(v))
    return np.array(v, dtype=np.float32)

class EffV2WithMeta(nn.Module):
    """
    EfficientNet-V2-M (ImageNet) + MLP metadata -> 2 clases (0=spoof, 1=live)
    """
    def __init__(self, meta_dim: int = META_DIM, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_v2_m(weights=weights)
        # La cabeza de torchvision v2 es classifier[1] (Linear) → in_features:
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_feat + 128, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 2)
        )

    def forward(self, x_img, x_meta):
        f = self.backbone(x_img)     # [B, in_feat]
        m = self.meta_mlp(x_meta)    # [B, 128]
        z = torch.cat([f, m], dim=1) # [B, in_feat+128]
        return self.classifier(z)
