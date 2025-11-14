from ultralytics import YOLO

model_pth = "/home/kevinteng/Desktop/dataset/trauma_thompson_dataset/task3_tools/exp_01/weights/best.pt"
model = YOLO(model_pth)
# Define path to video file
source = "/home/kevinteng/Desktop/dataset/trauma_thompson_dataset/task2_hands/train/videos/P05_05.mp4"

results = model.track(
    source=source,
    tracker="../task2_hand_tracking/bytetracker.yaml",
    save=True,
    save_txt=True,
    save_conf=True,
    save_crop=False,
)
