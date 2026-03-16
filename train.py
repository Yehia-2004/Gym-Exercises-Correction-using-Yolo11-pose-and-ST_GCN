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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        if epoch == 11:
            for p in model.backbone_model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.backbone_model.parameters(), 'lr': 1e-6, 'weight_decay': 0.05},
                {'params': model.fc1.parameters(), 'lr': 1e-4},
                {'params': model.fc2.parameters(), 'lr': 1e-4}
            ])

        for _, inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            

        train_loss = running_loss / len(train_loader)

        print(f"Train Loss: {train_loss}, Train Accuracy: {(correct/total) * 100.0}")


        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        best_val_loss = float('inf')
        for _, inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()  

        eval_loss = running_loss / len(val_loader)

        print(f"Validation Loss: {eval_loss}, Validation Accuracy: {(correct/total) * 100.0}\n\n")    

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model, "best_model.pth")

    return model

data_dir, yolo_version = config.PATHS.data_dir, config.PATHS.yolo_version
feature_extractor = Extractor(yolo_version)

aug_config = config.AUGMENTATION  
augmentor = Augmentor(aug_config.rot_degree, aug_config.resize_ratio, aug_config.bight_range, aug_config.probablity)

data_paths = list(Path(data_dir).rglob("*.mp4"))
split = int(0.9 * len(data_paths))
train_paths = data_paths[:split]
val_paths = data_paths[split:]

train_config = config.TRAINING
train_data = CorrectionData(train_paths, feature_extractor, augmentor)
val_data = CorrectionData(val_paths, feature_extractor)

train_loader = DataLoader(train_data, train_config.batch_size, True)
val_loader = DataLoader(val_data, train_config.batch_size, False)

backbone = ST_GCN(model_config.in_channels, model_config.hidden_channels, model_config.graph_args, True)
model = Model(model_config.num_classes, backbone)

train_model(model, train_loader, val_loader, model_config.device, float(train_config.lr), train_config.epochs)