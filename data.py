import numpy as np
from PIL import Image
import cv2
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CorrectionData(Dataset):
    def __init__(self, data_paths, target_frames=64):
        self.data = data_paths
        self.label_map = {
            "Correct sequence": 0,
            "Sagging hips": 1,
            "Partial range": 2,
            "Piked hips": 3,
            "Flared elbows": 4,
            "Original wrong": 5,
            "Wrong sequence": 5 # fallback for old structures
        }
        self.target_frames = target_frames

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        npy_path = self.data[i]
        
        label_name = npy_path.parent.name
        video_name = f"{label_name}/{npy_path.stem}" 
        encoded_label = self.label_map.get(label_name, 5) # Default to 5 if unknown
        encoded_label = torch.tensor(encoded_label, dtype=torch.long)

        features = np.load(npy_path)['features'] # Load from npz [Frames, 17, 3]
        features = torch.tensor(features, dtype=torch.float32)
        
        # SGN processing and reshaping
        # Desired shape for interpolation: [3, 17, Frames]
        features = features.permute(2, 1, 0) # [3, 17, Frames]
        
        # Interpolate temporally to fixed target_frames
        # F.interpolate expects [batch, channels, length] so we reshape to [17*3, Frames]
        c, v, t = features.shape
        features = features.reshape(1, c*v, t)
        features = F.interpolate(features, size=self.target_frames, mode='linear', align_corners=False)
        features = features.reshape(c, v, self.target_frames) # [3, 17, target_frames]
        
        # Rearrange to ST-GCN compatible shape [C, T, V, M] (our extract_feature expects this input)
        features = features.permute(0, 2, 1).unsqueeze(-1) # -> [3, target_frames, 17, 1]

        return video_name, features, encoded_label