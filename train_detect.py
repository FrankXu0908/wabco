from ultralytics import YOLO

model = YOLO("weights/pretrained/yolov8m.pt")
# Train the model on the own dataset for 50 epochs
train_results = model.train(
    data="train.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=[0,1],  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    batch=32  # Batch size
)
