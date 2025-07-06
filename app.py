from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

# Load the YOLO model for object detection
model = YOLO("runs/detect/train3/weights/best.pt")
# Define the directory containing images for evaluation
eval_img_dir = Path("images")

# Ensure the results directory exists
result_dir = Path("results")
result_dir.mkdir(exist_ok=True)

# Create a directory for cropped images
crop_dir = result_dir / "crops"
crop_dir.mkdir(exist_ok=True)

def detect_and_crop_images(img_path:str, 
                           eval_img_dir=eval_img_dir,
                           model=model, 
                           crop_dir=crop_dir,
                           result_dir=result_dir) -> list:
    """
    Detect 6 objects in an image, sort them by position,
    and return a list of 6 cropped image arrays.
    If less than 6 are detected, estimate missing ones based on spacing. 
    """
    eval_img_path = eval_img_dir / img_path
    if not eval_img_path.exists():
        raise FileNotFoundError(f"Image path {eval_img_path} does not exist.")

    img_name_without_ext = eval_img_path.stem
    results = model.predict(eval_img_path, save=True, project=result_dir, exist_ok=True)[0]
    boxes = results.boxes
    image = cv2.imread(str(eval_img_path))
    #获取图片长宽便于判断
    h, w = image.shape[:2]
    if len(boxes) != 6:
        print(f"⚠️ Warning: {img_name_without_ext} - Detected {len(boxes)} boxes (expecting 6)")
        """如果检测到的框数量不等于6，需要通过两个中心点物理位置几乎相等的特性，来找出哪个位置缺失，
        并且将缺失的框补齐。 以下代码来自ChatGPT，需要测试后使用
        Step 1: Detect Bounding Boxes → Get Centers
        detected_boxes = [box.xyxy[0].cpu().numpy() for box in results.boxes]
        centers = [((x1+x2)/2, (y1+y2)/2) for (x1, y1, x2, y2) in detected_boxes]
        Step 2: Determine Layout Direction
        range_x = max(x for x, y in centers) - min(x for x, y in centers)
        range_y = max(y for x, y in centers) - min(y for x, y in centers)
        horizontal = range_x > range_y
        Step 3: Sort Centers Left→Right or Top→Bottom
        sorted_indices = sorted(range(len(centers)), key=lambda i: centers[i][0] if horizontal else centers[i][1])
        centers_sorted = [centers[i] for i in sorted_indices]
        boxes_sorted = [detected_boxes[i] for i in sorted_indices]
        Step 4: Compute Spacing and Estimate Missing
        import numpy as np

        # Get 1D positions
        positions = [x if horizontal else y for x, y in centers_sorted]

        # Calculate spacing
        spacings = np.diff(positions)
        median_spacing = np.median(spacings)

        # Insert estimated missing boxes
        final_boxes = []
        for i in range(len(boxes_sorted) + 1):
            if i == 0:
                final_boxes.append(boxes_sorted[0])
                continue
            if i == len(boxes_sorted):
                final_boxes.append(boxes_sorted[-1])
                continue

            # Check if spacing too large → likely a missing box
            gap = positions[i] - positions[i - 1]
            if gap > 1.5 * median_spacing:
                # Estimate missing box center
                estimated_center = (positions[i - 1] + median_spacing, centers_sorted[i - 1][1]) if horizontal else \
                                (centers_sorted[i - 1][0], positions[i - 1] + median_spacing)

                # Estimate box size from previous box
                prev_box = boxes_sorted[i - 1]
                box_w = prev_box[2] - prev_box[0]
                box_h = prev_box[3] - prev_box[1]

                x_c, y_c = estimated_center
                x1 = int(x_c - box_w / 2)
                x2 = int(x_c + box_w / 2)
                y1 = int(y_c - box_h / 2)
                y2 = int(y_c + box_h / 2)

                # Insert estimated box
                final_boxes.append(np.array([x1, y1, x2, y2]))
            final_boxes.append(boxes_sorted[i])
        """
    # Extract XYXY and sort by horizontal (left to right) position
    xyxy_list = [box.xyxy[0].cpu().numpy() for box in boxes]
    
    # Decide sorting axis
    if w >= h:
        # Landscape → sort left-to-right (x1)
        xyxy_sorted = sorted(xyxy_list, key=lambda b: b[0])
    else:
        # Portrait → sort top-to-bottom (y1)
        xyxy_sorted = sorted(xyxy_list, key=lambda b: b[1])

    # Crop and return
    crops = [] 
    for idx, box in enumerate(xyxy_sorted, start=1):
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        save_path =  crop_dir / f"{img_name_without_ext}_{idx}.jpg"
        cv2.imwrite(save_path, crop)
        crops.append(crop)
    return crops


# Load the YOLO model for classification
#classifier = YOLO("runs/classify/train/weights/best.pt")
def classify_crops(crops: list, classifier, transform, class_labels=("OK", "NG")) -> list:
    """
    Given a list of cropped images, run classification and return a list of predicted labels.
    """

    results = []
    for crop in crops:
        if crop is None or crop.size == 0:
            results.append("empty")
            continue
        input_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            output = classifier(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            results.append(class_labels[pred])
    return results


crops = detect_and_crop_images("8331751694019_.pic_hd.jpg")
print(f"Number of crops: {len(crops)}", crops)
