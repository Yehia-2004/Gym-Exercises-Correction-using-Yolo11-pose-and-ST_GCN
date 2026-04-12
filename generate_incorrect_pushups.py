from pathlib import Path
from ultralytics import YOLO
from Features_extraction import PushUpDistorter

correct_pushups_dir = Path(r"E:\PushUps")
correct_videos = list(correct_pushups_dir.glob("*.avi"))
print(len(correct_videos))

yolo = YOLO("yolo11m.pt")
distorter = PushUpDistorter()
for path in correct_videos:
    results = yolo(path, stream=True, verbose=False)
    distorter.apply
            