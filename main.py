import yaml
from easydict import EasyDict as edict

with open(r"config.yaml", 'r') as f:
    config = yaml.safe_load(f)
config = edict(config)

from pathlib import Path

from Features_extraction import Augmentor, Extractor
import torch
from torch.utils.data import DataLoader
from data import CorrectionData

from model import Model
from train import train_model

model_config = config.MODEL

from setup_repo import setup
repo_url = model_config.backbone_model
setup(Path(repo_url))
from net.st_gcn import Model as ST_GCN


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
