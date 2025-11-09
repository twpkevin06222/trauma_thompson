from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="./tools_det.yaml",
    epochs=50,
    optimizer="AdamW",
    lr0=0.001,
    imgsz=(540, 960),  # downsize by a factor of 2 due to VRAM constraints
    project="/home/kevinteng/Desktop/dataset/trauma_thompson_dataset/task3_tools",
    name="exp_01",
)
