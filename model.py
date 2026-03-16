import torch
from tqdm.auto import tqdm
from torch import nn, optim
import warnings
warnings.filterwarnings('ignore')

class Model(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone_model = backbone
        in_channels = self.backbone_model.fcn.in_channels
        
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(in_channels)
        )
        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(23040, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone_model.extract_feature(x)[0]
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
