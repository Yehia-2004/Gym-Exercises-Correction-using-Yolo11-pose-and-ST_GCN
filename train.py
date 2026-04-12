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

from sgn_model import SGNModel
from model import Model

def train_model(model, train_loader, val_loader, device, learning_rate, epochs=20):
    model.to(device)
    # 1. Update Criterion to CrossEntropy for Multi-class
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val_loss = float('inf')
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        model.train()
        running_loss, train_correct, total = 0.0, 0, 0
        
        if epoch == 11:
            for p in model.backbone_model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.backbone_model.parameters(), 'lr': 1e-6, 'weight_decay': 0.05},
                {'params': model.fc2.parameters(), 'lr': 1e-4}
            ])

        for _, inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            logits = model(inputs) # Shape: [bs, num_classes]
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Predict by argmax
            predicted = torch.argmax(logits, dim=1)
            train_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = train_correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for _, inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                logits = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(logits, dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        eval_loss = val_loss / val_total
        eval_acc = val_correct / val_total

        pbar.set_postfix({'T_L': f"{train_loss:.3f}", 'T_A': f"{train_acc:.2f}", 'V_L': f"{eval_loss:.3f}", 'V_A': f"{eval_acc:.2f}"})

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), "best_model.pth")

    return model

data_dir = config.PATHS.data_dir

all_data_paths = list(Path(data_dir).rglob("*.npz"))

# Isolate 'Original wrong' entirely to testing set as requested
test_only_paths = [p for p in all_data_paths if p.parent.name == "Original wrong"]
train_valid_paths = [p for p in all_data_paths if p.parent.name != "Original wrong"]

np.random.shuffle(train_valid_paths)

split = int(0.7 * len(train_valid_paths))
train_paths = train_valid_paths[:split]
val_paths = train_valid_paths[split:] + test_only_paths

train_config = config.TRAINING
train_data = CorrectionData(train_paths, target_frames=64)
val_data = CorrectionData(val_paths, target_frames=64)

train_loader = DataLoader(train_data, train_config.batch_size, True)
val_loader = DataLoader(val_data, train_config.batch_size, False)

backbone = SGNModel(num_classes=model_config.num_classes, seg=64)
for p in backbone.parameters():
  p.requires_grad = False

model = Model(model_config.num_classes, backbone)

train_model(model, train_loader, val_loader, model_config.device, float(train_config.lr), train_config.epochs)

