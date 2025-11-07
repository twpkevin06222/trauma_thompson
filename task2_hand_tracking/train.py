from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="hand_det.yaml",
    epochs=50,
    imgsz=(1280, 920),
    single_cls=True,
    project="/home/kevinteng/Desktop/dev/llm/personal_projects/healthcare/trauma_thompson/task2_hand_tracking",
    name="exp_01",
)
