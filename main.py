import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
from easydict import EasyDict as edict
from collections import deque
from ultralytics import YOLO

from sgn_model import SGNModel
from model import Model
from Features_extraction import Extractor

with open(r"config.yaml", 'r') as f:
    config = yaml.safe_load(f)
config = edict(config)

model_config = config.MODEL
yolo_version = config.PATHS.yolo_version

# Initialize Device and SGN Model
device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
backbone = SGNModel(num_classes=model_config.num_classes, seg=64)
action_model = Model(model_config.num_classes, backbone)
try:
    action_model.load_state_dict(torch.load("best_model.pth", map_location=device))
except Exception as e:
    print(f"Warning: Could not load trained weights: {e}. Model will use random weights.")
action_model.to(device)
action_model.eval()

extractor = Extractor(yolo_version)

capture_way = str(input("Upload an existing video: (V)\nUse the camera: (C)\n")).strip().upper()

def process_features(features_list):
    # Process sequential features into model's expected shape: [1, 3, 64, 17, 1]
    features = torch.tensor(np.array(features_list), dtype=torch.float32) # [Frames, 17, 3]
    features = features.permute(2, 1, 0) # [3, 17, Frames]
    
    # Interpolate temporally to fixed target_frames (64)
    c, v, t = features.shape
    features = features.reshape(1, c*v, t)
    features = F.interpolate(features, size=64, mode='linear', align_corners=False)
    features = features.reshape(1, c, v, 64) # [1, 3, 17, 64]
    features = features.permute(0, 1, 3, 2).unsqueeze(-1) # -> [1, 3, 64, 17, 1]
    return features.to(device)

feedback_dict = {
    0: ("Form: CORRECT (Great Job!)", (0, 255, 0)),
    1: ("Form: SAGGING HIPS (Keep core tight!)", (0, 0, 255)),
    2: ("Form: PARTIAL RANGE (Go deeper!)", (0, 165, 255)),
    3: ("Form: PIKED HIPS (Lower your hips!)", (0, 0, 255)),
    4: ("Form: FLARED ELBOWS (Tuck elbows close!)", (0, 0, 255)),
    5: ("Form: INCORRECT FORM DETECTED", (0, 0, 255))
}

if capture_way == "V":
    video_path = Path(input("Enter the Video's Path here: ").strip())
    raw_features = extractor.extract(video_path)
    features_tensor = process_features(raw_features)
    
    with torch.no_grad():
        logits = action_model(features_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        msg, _ = feedback_dict.get(predicted_class, feedback_dict[5])
        print(f"Prediction for entire video: {msg}")

elif capture_way == "C":
    yolo = YOLO(yolo_version)
    cap = cv2.VideoCapture(0)
    print("Camera running. Press 'q' to stop.")

    # State for Rep Counting and Sequences
    frame_buffer = deque(maxlen=64)
    state = "up"
    rep_count = 0
    feedback_msg = "Warming up..."
    feedback_color = (0, 255, 255) # Yellow initially

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo(frame, stream=False, verbose=False)
        r = results[0]
        
        frame_data = np.zeros((17, 3))
        if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy[0].shape[0] > 0:
            conf = r.keypoints.conf[0].reshape(-1, 1).cpu().numpy()
            norm_kpts = r.keypoints.xyn[0].cpu().numpy()
            frame_data = np.hstack([norm_kpts, conf])
            
            # Simple heuristic for Up/Down transition tracking Push-ups
            # L_SH=5, R_SH=6, L_EL=7, R_EL=8
            avg_shoulder_y = (norm_kpts[5, 1] + norm_kpts[6, 1]) / 2.0
            avg_elbow_y = (norm_kpts[7, 1] + norm_kpts[8, 1]) / 2.0
            
            is_down = avg_shoulder_y > avg_elbow_y - 0.05
            is_up = avg_shoulder_y < avg_elbow_y - 0.15

            if state == "up" and is_down:
                state = "down"
            elif state == "down" and is_up:
                state = "up"
                rep_count += 1
                
        frame_buffer.append(frame_data)
        
        # Real-time feedback window
        if len(frame_buffer) == 64:
            features_t = process_features(list(frame_buffer))
            with torch.no_grad():
                logits = action_model(features_t)
                predicted_class = torch.argmax(logits, dim=1).item()
                
            feedback_msg, feedback_color = feedback_dict.get(predicted_class, feedback_dict[5])

        # Draw overlays
        cv2.putText(frame, f"Reps: {rep_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, feedback_msg, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color, 2)
        cv2.putText(frame, f"State: {state.upper()}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("Push-up Tracker", frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()