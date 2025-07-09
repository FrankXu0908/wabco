import os
import cv2
import time
import threading
import torch
import onnxruntime as ort
import gradio as gr
import logging
from detector import ObjectDetector, DefectClassifier
import queue
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Global shared variables
latest_frame = None
lock = threading.Lock()

# Queues for inter-thread communication
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

# Directory for NG crops
ng_crop_dir = "ng_crops"
os.makedirs(ng_crop_dir, exist_ok=True)


# Add to imports
from detector import ObjectDetector, DefectClassifier

