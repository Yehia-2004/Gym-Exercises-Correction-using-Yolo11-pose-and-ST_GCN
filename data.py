import numpy as np
from PIL import Image
import cv2
from pathlib import Path

import torch
from torch.utils.data import Dataset

from Features_extraction import Augmentor, Extractor

class CorrectionData(Dataset):
    def __init__(self, data: list, feature_extractor: Extractor, augmentor: Augmentor = None):
        self.data = data
        self.label_map = {"Correct sequence": 1, "Wrong sequence": 0}

        self.extractor = feature_extractor
        self.augmentor = augmentor

    def __len__(self): return len(self.data)

    def process_video_path(self, video_path, target_frames=30):
        cap = cv2.VideoCapture(str(video_path))
        
        # 1. Get total frame count (e.g., 125 or 200)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 2. Calculate 30 evenly spaced indices
        # np.linspace ensures we pick the first frame (0) and the last frame (total-1)
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        
        frames = []
        
        for idx in indices:
            # 3. Jump directly to the specific frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            
            if success:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                raise NotImplementedError("Zeby")

        cap.release()

        return frames

    def __getitem__(self, i):
        video_path = self.data[i]
        frames = self.process_video_path(video_path)
        if self.augmentor:
            frames = self.augmentor.transform(frames)

        features = self.extractor.process(frames)
        
        label_name = video_path.parent.name
        video_name = f"{label_name}/{video_path.stem}" 
        encoded_label = self.label_map.get(label_name, 0)

        # Shape: (Frames, Joints, Channels) -> (Channels, Frames, Joints)
        features = features.permute(2,0,1).unsqueeze(-1).float()
        
        return video_name, features, encoded_label