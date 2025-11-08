from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="./hand_det.yaml",
    epochs=20,
    optimizer="AdamW",
    lr0=0.001,
    mosaic=0.0,
    single_cls=True,
    imgsz=(920, 1280),
    project="/home/kevinteng/Desktop/dataset/trauma_thompson_dataset/task2_hands",
    name="exp_01",
)
