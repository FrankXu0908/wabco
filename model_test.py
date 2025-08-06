import os
import time
import torch
import onnxruntime as ort
from detector import ObjectDetector_frame, DefectClassifier
import cv2

def main():
    num_cpu_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(num_cpu_threads)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_cpu_threads  # Controls ONNX model thread use
    # Initialize the object detector
    detector = ObjectDetector_frame()
    # Initialize the defect classifier
    classifier = DefectClassifier()
    
    # Detect and crop images
    start_time = time.time()
    img_path = "./images/cam0_1_NG_20250731_225018.jpg"  # "1771344.jpg" "8331751694019_.pic_hd.jpg"
    frame = cv2.imread(img_path)
    crops = detector.detect_and_crop_images(frame,1,"8-6")
    print(f"Number of crops: {len(crops)}, time taken:{time.time() - start_time:.2f}seconds")

    # Classify the cropped images
    start_time = time.time()
    predictions = classifier.classify(crops,1, "8-6")
    print(predictions, f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()