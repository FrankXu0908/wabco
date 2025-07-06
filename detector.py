from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import torchvision
from ultralytics import YOLO
import torch
from torchvision import models, transforms

class ObjectDetector:
    def __init__(self,
                 yolo_weights_path="weights/detector/best.pt",
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

class DefectClassifier:
    """
    Given a list of cropped images, run classification and return a dictionary of indices and predicted labels.
    """
    def __init__(self, model_path=None, num_classes=2, class_labels=("OK", "NG")):
        self.device = torch.device("cpu")
        self.class_labels = class_labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.model = models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()

    def classify(self, crops: list) -> list:
        tensors = []
        valid_indices = []
        for idx, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            tensor = self.transform(crop)
            tensors.append(tensor)
            valid_indices.append(idx)

        if not tensors:
            return ["empty"] * len(crops)

        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            outputs = self.model(batch)
            preds = torch.argmax(outputs, dim=1).tolist()

        # Fill out results in original order
        results = []
        j = 0
        for i in range(len(crops)):
            if i in valid_indices:
                results.append({i + 1: self.class_labels[preds[j]]})
                j += 1
            else:
                results.append({i + 1: "empty"})
        return results

    # def classify(self, crops: list) -> list:
    #     results = []
    #     with torch.inference_mode():
    #         for idx, crop in enumerate(crops, start=1):
    #             if crop is None or crop.size == 0:
    #                 results.append("empty")
    #                 continue
    #             input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
    #             output = self.model(input_tensor)
    #             pred = torch.argmax(output, dim=1).item()
    #             results.append({idx: self.class_labels[pred]})
    #     return results