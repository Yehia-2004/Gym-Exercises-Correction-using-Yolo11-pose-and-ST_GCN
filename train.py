import numpy as np
from pathlib import Path
import yaml
from easydict import EasyDict as edict
from tqdm.auto import tqdm

from Features_extraction import Augmentor, Extractor
from data import CorrectionData

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import warnings
warnings.filterwarnings('ignore')

with open(r"config.yaml", 'r') as f:
    config = yaml.safe_load(f)
config = edict(config)

model_config = config.MODEL

from setup_repo import setup
repo_url = model_config.backbone_model
setup(Path(repo_url))
from net.st_gcn import Model as ST_GCN

from model import Model

def train_model(model, train_loader, val_loader, device, learning_rate, epochs=20):
    model.to(device)
    # 1. Update Criterion
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val_loss = float('inf')
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        model.train()
        running_loss, train_correct, total = 0.0, 0, 0
        
        # Fine-tuning logic
        if epoch == 11:
            for p in model.backbone_model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.backbone_model.parameters(), 'lr': 1e-6, 'weight_decay': 0.05},
                {'params': model.fc1.parameters(), 'lr': 1e-4},
                {'params': model.fc2.parameters(), 'lr': 1e-4}
            ])

        for _, inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            logits = model(inputs).squeeze() # These are now raw logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # 2. Accuracy requires manual sigmoid for the threshold
            probs = torch.sigmoid(logits) 
            predicted = (probs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = train_correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for _, inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                logits = model(inputs).squeeze()
                loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        eval_loss = val_loss / val_total
        eval_acc = val_correct / val_total

        pbar.set_postfix({'T_L': f"{train_loss:.3f}", 'T_A': f"{train_acc:.2f}", 'V_L': f"{eval_loss:.3f}", 'V_A': f"{eval_acc:.2f}"})

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), "best_model.pth")

    return model

data_dir, yolo_version = config.PATHS.data_dir, config.PATHS.yolo_version
feature_extractor = Extractor(yolo_version)

aug_config = config.AUGMENTATION  
augmentor = Augmentor(aug_config.rot_degree, aug_config.resize_ratio, aug_config.bight_range, aug_config.probablity)

data_paths = list(Path(data_dir).rglob("*.mp4"))
np.random.shuffle(data_paths)

split = int(0.7 * len(data_paths))
train_paths = data_paths[:split]
val_paths = data_paths[split:]

train_config = config.TRAINING
train_data = CorrectionData(train_paths, feature_extractor, augmentor)
val_data = CorrectionData(val_paths, feature_extractor)

train_loader = DataLoader(train_data, train_config.batch_size, True)
val_loader = DataLoader(val_data, train_config.batch_size, False)

backbone = ST_GCN(model_config.in_channels, model_config.hidden_channels, model_config.graph_args, True)
for p in backbone.parameters():
  p.requires_grad = False

model = Model(model_config.num_classes, backbone)

train_model(model, train_loader, val_loader, model_config.device, float(train_config.lr), train_config.epochs)
