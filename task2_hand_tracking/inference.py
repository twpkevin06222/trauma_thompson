from ultralytics import YOLO
from glob import glob
import os
import pandas as pd

# load the csv file
csv_pth = "/home/kevinteng/Desktop/dev/llm/personal_projects/healthcare/trauma_thompson/task2_hand_tracking/outputs/info_df.csv"
df = pd.read_csv(csv_pth)
val_video_ids = df[df["split"] == "val"]["video_id"].tolist()
# load the model
model_pth = "/home/kevinteng/Desktop/dataset/trauma_thompson_dataset/task2_hands/exp_01/weights/best.pt"
video_dir = (
    "/home/kevinteng/Desktop/dataset/trauma_thompson_dataset/task2_hands/train/videos"
)
model = YOLO(model_pth)

for video_pth in glob(os.path.join(video_dir, "*.mp4")):
    video_id = os.path.basename(video_pth).replace(".mp4", "")
    if video_id not in val_video_ids:
        print(f"Skipping video: {video_pth} due to being in the training set")
        continue
    print(f"Processing video: {video_pth}")
    results = model.track(
        source=video_pth,
        tracker="bytetracker.yaml",
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=False,
    )
    print(f"Finished processing video: {video_pth}")
    print("-" * 100)
    print()
