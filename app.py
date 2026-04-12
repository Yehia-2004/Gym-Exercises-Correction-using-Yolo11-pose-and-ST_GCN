import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
from easydict import EasyDict as edict
from collections import deque
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify

from sgn_model import SGNModel
from model import Model

app = Flask(__name__)

app_state = {
    "reps": 0,
    "feedback_msg": "Initializing Model...",
    "state": "UP",
    "color_class": "status-neutral"
}

with open(r"config.yaml", 'r') as f:
    config = yaml.safe_load(f)
config = edict(config)

model_config = config.MODEL
yolo_version = config.PATHS.yolo_version

device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
backbone = SGNModel(num_classes=model_config.num_classes, seg=64)
action_model = Model(model_config.num_classes, backbone)

try:
    action_model.load_state_dict(torch.load("best_model.pth", map_location=device))
except Exception as e:
    print(f"Warning: Could not load weights: {e}")
action_model.to(device)
action_model.eval()

yolo = YOLO(yolo_version)

feedback_dict = {
    0: ("Form: CORRECT (Great Job!)", "status-good"),
    1: ("Form: SAGGING HIPS (Keep core tight!)", "status-bad"),
    2: ("Form: PARTIAL RANGE (Go deeper!)", "status-warn"),
    3: ("Form: PIKED HIPS (Lower your hips!)", "status-bad"),
    4: ("Form: FLARED ELBOWS (Tuck elbows close!)", "status-bad"),
    5: ("Form: INCORRECT FORM DETECTED", "status-bad")
}

def process_features(features_list):
    features = torch.tensor(np.array(features_list), dtype=torch.float32)
    features = features.permute(2, 1, 0)
    c, v, t = features.shape
    features = features.reshape(1, c*v, t)
    features = F.interpolate(features, size=64, mode='linear', align_corners=False)
    features = features.reshape(1, c, v, 64)
    features = features.permute(0, 1, 3, 2).unsqueeze(-1)
    return features.to(device)

def generate_frames():
    global app_state
    cap = cv2.VideoCapture(0)
    
    frame_buffer = deque(maxlen=64)
    local_state = "up"
    rep_count = 0
    
    app_state["feedback_msg"] = "Warming up Camera..."
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = yolo(frame, stream=False, verbose=False)
        r = results[0]
        
        frame_data = np.zeros((17, 3))
        if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy[0].shape[0] > 0:
            conf = r.keypoints.conf[0].reshape(-1, 1).cpu().numpy()
            norm_kpts = r.keypoints.xyn[0].cpu().numpy()
            frame_data = np.hstack([norm_kpts, conf])
            
            avg_shoulder_y = (norm_kpts[5, 1] + norm_kpts[6, 1]) / 2.0
            avg_elbow_y = (norm_kpts[7, 1] + norm_kpts[8, 1]) / 2.0
            
            is_down = avg_shoulder_y > avg_elbow_y - 0.05
            is_up = avg_shoulder_y < avg_elbow_y - 0.15

            if local_state == "up" and is_down:
                local_state = "down"
            elif local_state == "down" and is_up:
                local_state = "up"
                rep_count += 1
                
        frame_buffer.append(frame_data)
        
        if len(frame_buffer) == 64:
            features_t = process_features(list(frame_buffer))
            with torch.no_grad():
                logits = action_model(features_t)
                predicted_class = torch.argmax(logits, dim=1).item()
                
            msg, color = feedback_dict.get(predicted_class, feedback_dict[5])
            app_state["feedback_msg"] = msg
            app_state["color_class"] = color
            
        app_state["reps"] = rep_count
        app_state["state"] = local_state.upper()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(app_state)

if __name__ == '__main__':
    app.run(port=5000, threaded=True)
