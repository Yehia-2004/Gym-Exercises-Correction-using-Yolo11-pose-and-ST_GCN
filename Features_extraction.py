import numpy as np
from PIL import Image

from pathlib import Path
import cv2
from dataclasses import dataclass
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

from vidaug import augmentors as va 

from ultralytics import YOLO
import torch

class Augmentor:
    def __init__(self, rot_degree, resize_ratio, bright_percent, probability):
        self.degree = rot_degree
        self.resize = resize_ratio
        self.bright = bright_percent
        self.p = probability

    
    def transform(self, frames):
        seq = va.Sequential([
                va.RandomRotate(degrees=self.degree), 
                va.Sometimes(self.p, va.HorizontalFlip()),
                va.RandomResize(self.resize),
                va.Sometimes(self.p, va.Multiply(self.bright[0])),
                va.Sometimes(self.p, va.Multiply(self.bright[1])),
                # va.Sometimes(self.p , va.Salt(0.1)),
                # va.Sometimes(self.p , va.Pepper(0.1)),
                va.TemporalElasticTransformation()
            ])
        
        return seq(frames) 

class Extractor:
    def __init__(self, yolo_version: YOLO):
        self.yolo = YOLO(yolo_version)

    def process(self, frames):
        features = []
        results = self.yolo(frames, stream=True, verbose=False)
        
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy[0].shape[0] > 0:
                conf = r.keypoints.conf[0].reshape(-1, 1)
                norm_kpts = r.keypoints.xyn[0].cpu().numpy()
                conf_np = conf.cpu().numpy()
                frame_data = np.hstack([norm_kpts, conf_np]) 
                features.append(frame_data)
            else:
                features.append(np.zeros((17, 3)))

        # label = video_path.parent.stem
        # current_out_dir = self.out_dir / label 
        # current_out_dir.mkdir(parents=True, exist_ok=True)
        # out_path = current_out_dir / f"{video_path.stem}.npy"

        # np.save(out_path, features_matrix)
        return torch.tensor(features)

    # def process_all(self):
    #     all_features = []
    #     for video_path in tqdm(self.data, desc="Extracting Features"):
    #         extracted_features = self.extract_save_features(video_path)
    #         all_features.append(extracted_features)

    #     return torch.tensor(all_features, torch.float32)