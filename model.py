import torch
from tqdm.auto import tqdm
from torch import nn, optim
import warnings
warnings.filterwarnings('ignore')

class Model(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone_model = backbone
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # SGN returns a flattened feature vector [bs, 512]
        x = self.backbone_model.extract_feature(x)
        x = self.fc2(x)
        return x
