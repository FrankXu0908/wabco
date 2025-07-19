# from PIL import Image
# import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import torchvision
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import onnxruntime as ort
import numpy as np
import time
import logging

class ObjectDetector:
    def __init__(self,
                 yolo_weights_path="weights/detector/detector_v1.pt",
                 eval_img_dir=Path("images"),
                 result_dir=Path("results")):
        self.model = YOLO(yolo_weights_path)
        self.eval_img_dir = eval_img_dir
        self.result_dir = result_dir
        self.result_dir.mkdir(exist_ok=True)
        self.crop_dir = self.result_dir / "crops"
        self.crop_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def detect_and_crop_images(self, img_path: str) -> list:
        eval_img_path = self.eval_img_dir / img_path
        if not eval_img_path.exists():
            raise FileNotFoundError(f"Image path {eval_img_path} does not exist.")
        img_name_without_ext = eval_img_path.stem
        results = self.model.predict(eval_img_path, save=True, project=self.result_dir, exist_ok=True,conf=0.4)[0]
        boxes = results.boxes
        image = cv2.imread(str(eval_img_path))
        h, w = image.shape[:2]
        if len(boxes) != 6:
            print(f"⚠️ Warning: {img_name_without_ext} - Detected {len(boxes)} boxes (expecting 6)")
            # Handle the case where less than 6 boxes are detected
            # This part can be customized based on your specific requirements
        xyxy_list = [box.xyxy[0].cpu().numpy() for box in boxes]
        if w >= h:
            xyxy_sorted = sorted(xyxy_list, key=lambda b: b[0])
        else:
            xyxy_sorted = sorted(xyxy_list, key=lambda b: b[1])
        crops = []
        for idx, box in enumerate(xyxy_sorted, start=1):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            save_path =  self.crop_dir / f"{img_name_without_ext}_{idx}.bmp"
            cv2.imwrite(save_path, crop)
            crops.append(crop)
        return crops

class DefectClassifier:
    """
    Given a list of cropped images, run classification using an ONNX model
    and return a dictionary of indices and predicted labels.
    """
    def __init__(self, onnx_path="weights/biclassifier/efficientnet_b1_trained.onnx", class_labels=("NG","OK")):
        self.class_labels = class_labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to RGB
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    def classify(self, crops: list, camera_id) -> list:
        tensors = []
        print(f"Classifying {len(crops)} crops for camera {camera_id}")
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            tensor = self.transform(crop).unsqueeze(0).numpy()
            tensors.append(tensor)

        batch = np.vstack(tensors).astype(np.float32)
        outputs = self.session.run(None, {"input": batch})
        preds = np.argmax(outputs[0], axis=1)

        # Fill out results in original order
        results = []
        for i in range(len(crops)):
            results.append({i + 1: self.class_labels[preds[i]]})

        # save each crop with its prediction
        result_path = Path("results")    
        result_ok = result_path / "OK"
        result_ng = result_path / "NG"
        result_path.mkdir(exist_ok=True)
        result_ok.mkdir(exist_ok=True)
        result_ng.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        for idx, crop in enumerate(crops):
            if crop is not None:
                if preds[idx] == 0:
                    save_path = result_ng / f"cam{camera_id}_{idx + 1}_NG_{timestamp}.jpg"
                else:
                    save_path = result_ok / f"cam{camera_id}_{idx + 1}_OK_{timestamp}.jpg"
                cv2.imwrite(str(save_path), crop)
        return results, preds
