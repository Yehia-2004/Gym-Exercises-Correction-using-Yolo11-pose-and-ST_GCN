from pathlib import Path
import yaml
from easydict import EasyDict as edict

from Features_extraction import Extractor

with open(r"config.yaml", 'r') as f:
    config = yaml.safe_load(f)
config = edict(config)

yolo_version = config.PATHS.yolo_version
extractor = Extractor(yolo_version)

video_path = r"E:\PushUps\v_PushUps_g01_c01.avi"

features = extractor.process(video_path)
print(features)