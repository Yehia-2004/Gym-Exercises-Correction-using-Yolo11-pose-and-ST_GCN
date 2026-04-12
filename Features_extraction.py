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

    def extract(self, path):
        features = []
        results = self.yolo(path, stream=True, verbose=False)
        
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

    class PushUpDistorter:
        def __init__(self, keypoint_format="COCO"):
            # COCO indices: Shoulder(5,6), Elbow(7,8), Wrist(9,10), Hip(11,12), Ankle(15,16)
            self.L_SH = 5; self.R_SH = 6
            self.L_HI = 11; self.R_HI = 12
            self.L_AN = 15; self.R_AN = 16

        def apply_sagging_hips(self, sequence, intensity=0.15):
            """Pushes the hips 'down' relative to the shoulder-ankle line."""
            distorted = sequence.copy()
            for f in range(len(distorted)):
                # Calculate average body length (Shoulder to Ankle) for scaling
                body_len = np.linalg.norm(distorted[f, self.L_SH] - distorted[f, self.L_AN])
                
                # Shift both hips down (Y-axis)
                distorted[f, [self.L_HI, self.R_HI], 1] += body_len * intensity
            return distorted

        def apply_partial_range(self, sequence, limit_factor=0.5):
            """Restricts vertical movement of the shoulders to simulate half-reps."""
            distorted = sequence.copy()
            y_coords = distorted[:, [self.L_SH, self.R_SH], 1]
            y_min = np.min(y_coords) # Highest point of rep
            
            # Scale the distance from the top so it never reaches the bottom
            for f in range(len(distorted)):
                distorted[f, [self.L_SH, self.R_SH], 1] = y_min + (distorted[f, [self.L_SH, self.R_SH], 1] - y_min) * limit_factor
            return distorted

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm.auto import tqdm

# ... (Your Augmentor and Extractor classes remain the same) ...

class PushUpDistorter:
    def __init__(self):
        # COCO indices for YOLOv11
        self.L_SH = 5; self.R_SH = 6
        self.L_EL = 7; self.R_EL = 8
        self.L_HI = 11; self.R_HI = 12
        self.L_AN = 15; self.R_AN = 16

    def apply_sagging_hips(self, sequence, intensity=0.12):
        distorted = sequence.copy()
        for f in range(len(distorted)):
            body_len = np.linalg.norm(distorted[f, self.L_SH, :2] - distorted[f, self.L_AN, :2])
            distorted[f, [self.L_HI, self.R_HI], 1] += body_len * intensity # Y increases (down)
        return distorted

    def apply_partial_range(self, sequence, limit_factor=0.6):
        distorted = sequence.copy()
        y_sh = distorted[:, [self.L_SH, self.R_SH], 1]
        y_min = np.min(y_sh) # highest point
        
        for f in range(len(distorted)):
            distorted[f, [self.L_SH, self.R_SH], 1] = y_min + (distorted[f, [self.L_SH, self.R_SH], 1] - y_min) * limit_factor
        return distorted

    def apply_piked_hips(self, sequence, intensity=0.15):
        distorted = sequence.copy()
        for f in range(len(distorted)):
            body_len = np.linalg.norm(distorted[f, self.L_SH, :2] - distorted[f, self.L_AN, :2])
            distorted[f, [self.L_HI, self.R_HI], 1] -= body_len * intensity # Y decreases (up)
        return distorted

    def apply_flared_elbows(self, sequence, intensity=0.15):
        distorted = sequence.copy()
        for f in range(len(distorted)):
            # Distort elbows wider out (Left goes more left, Right goes more right)
            distorted[f, self.L_EL, 0] -= intensity
            distorted[f, self.R_EL, 0] += intensity
        return distorted


correct_dir = Path(r"D:\GP\PushUps_Dataset\Correct sequence")
wrong_dir = Path(r"D:\GP\PushUps_Dataset\Wrong sequence")

output_base = Path("NPY_Features")

# Setup individual output directories for classes
out_dirs = {
    "Correct sequence": output_base / "Correct sequence",
    "Sagging hips": output_base / "Sagging hips",
    "Partial range": output_base / "Partial range",
    "Piked hips": output_base / "Piked hips",
    "Flared elbows": output_base / "Flared elbows",
    "Original wrong": output_base / "Original wrong"
}

for d in out_dirs.values():
    d.mkdir(parents=True, exist_ok=True)

yolo_model = YOLO("yolo11m-pose.pt")
extractor = Extractor(yolo_model)
distorter = PushUpDistorter()


for path in tqdm(list(correct_dir.glob("*")), desc="Processing Correct Videos & Synthesizing Distortions"):  
    raw_features = np.array(extractor.extract(path)) 

    # 1. Save Original Correct
    np.savez_compressed(out_dirs["Correct sequence"] / f"{path.stem}.npz", features=raw_features)

    # 2. Sagging Hips
    sagging = distorter.apply_sagging_hips(raw_features, intensity=0.15)
    np.savez_compressed(out_dirs["Sagging hips"] / f"{path.stem}_sagging.npz", features=sagging)
    
    # 3. Partial Range
    partial = distorter.apply_partial_range(raw_features, limit_factor=0.5)
    np.savez_compressed(out_dirs["Partial range"] / f"{path.stem}_partial.npz", features=partial)

    # 4. Piked Hips
    piked = distorter.apply_piked_hips(raw_features, intensity=0.15)
    np.savez_compressed(out_dirs["Piked hips"] / f"{path.stem}_piked.npz", features=piked)

    # 5. Flared Elbows
    flared = distorter.apply_flared_elbows(raw_features, intensity=0.15)
    np.savez_compressed(out_dirs["Flared elbows"] / f"{path.stem}_flared.npz", features=flared)

for path in tqdm(list(wrong_dir.glob("*")), desc="Processing Original Wrong Videos"):  
    raw_features = np.array(extractor.extract(path)) 

    if not raw_features.any() or len(raw_features) == 0:
        print(f"{path} doesn't have any valid frames")
        continue

    # Save original wrong manually recorded examples
    np.savez_compressed(out_dirs["Original wrong"] / f"{path.stem}_orig.npz", features=raw_features)