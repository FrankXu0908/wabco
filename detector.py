from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self,
                 yolo_weights_path="runs/detect/train3/weights/best.pt",
                 eval_img_dir=Path("images"),
                 result_dir=Path("results")):
        self.model = YOLO(yolo_weights_path)
        self.eval_img_dir = eval_img_dir
        self.result_dir = result_dir
        self.result_dir.mkdir(exist_ok=True)
        self.crop_dir = self.result_dir / "crops"
        self.crop_dir.mkdir(exist_ok=True)
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
            save_path =  self.crop_dir / f"{img_name_without_ext}_{idx}.jpg"
            cv2.imwrite(save_path, crop)
            crops.append(crop)
        return crops