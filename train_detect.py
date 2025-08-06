from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor

def train_model1():
    model1 = YOLO("weights/pretrained/yolov8m.pt")
    model1.train(
        data="train.yaml",
        epochs=200,
        imgsz=640,
        device=0,  # assign to GPU 0
        batch=32
    )

def train_model2():
    model2 = YOLO("weights/pretrained/yolov8n.pt")
    model2.train(
        data="train.yaml",
        epochs=200,
        imgsz=640,
        device=1,  # assign to GPU 1
        batch=32
    )

def train_yolo_model():
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(train_model1),
            executor.submit(train_model2)
        ]
        for future in futures:
            future.result()

train_yolo_model()
