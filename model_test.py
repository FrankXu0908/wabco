import os
import time
import torch
import onnxruntime as ort
from detector import ObjectDetector, DefectClassifier

def main():
    num_cpu_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(num_cpu_threads)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_cpu_threads  # Controls ONNX model thread use
    # Initialize the object detector
    detector = ObjectDetector()
    # Initialize the defect classifier
    classifier = DefectClassifier()
    
    # Detect and crop images
    img_path = "image_20250711_ng1.bmp"  # "1771344.jpg" "8331751694019_.pic_hd.jpg"
    crops = detector.detect_and_crop_images(img_path)

    # Classify the cropped images
    start_time = time.time()
    predictions = classifier.classify(crops)
    print(f"Number of crops: {len(crops)}")
    print(predictions, f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()