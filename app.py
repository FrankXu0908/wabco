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


# Initialize models
num_cpu_threads = min(4, os.cpu_count() or 1)
torch.set_num_threads(num_cpu_threads)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = num_cpu_threads
detector = ObjectDetector()
classifier = DefectClassifier()

def wait_for_plc_trigger():
    # Placeholder for GPIO or serial/PLC signal monitoring
    time.sleep(0.5)
    return True

def camera_capture(camera_id=0):
    global latest_frame
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logging.error("Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame.copy()
            # Put frame into frame_queue for processing
            try:
                frame_queue.put(frame.copy(), timeout=0.1)
            except queue.Full:
                pass  # Drop frame if queue is full
        else:
            logging.warning("Failed to read frame from camera.")
        time.sleep(0.1)

def send_signal_to_plc(ng_positions):
    if ng_positions:
        logging.info(f"\U0001F4E1 Sending reject signal to PLC for positions: {ng_positions}")
    else:
        logging.info("\u2705 All parts OK, no signal sent to PLC.")

# Threaded defect processing
def defect_processing_thread():
    while True:
        frame = frame_queue.get()
        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, frame)

        crops = detector.detect_and_crop_images(temp_path)
        start = time.time()
        predictions = classifier.classify(crops)
        elapsed = time.time() - start

        logging.info(f"\U0001F50D Classification done in {elapsed:.2f}s: {predictions}")

        ng_positions = []
        for i, (crop, pred_dict) in enumerate(zip(crops, predictions)):
            label = list(pred_dict.values())[0]
            if label == "NG":
                ng_positions.append(i + 1)
                filename = f"{uuid.uuid4().hex[:8]}_pos{i+1}.jpg"
                cv2.imwrite(os.path.join(ng_crop_dir, filename), crop)

        send_signal_to_plc(ng_positions)
        try:
            result_queue.put((frame, predictions), timeout=0.1)
        except queue.Full:
            pass  # Drop if result queue is full

def gradio_interface():
    def display_latest_result():
        if result_queue.empty():
            return None, "No result yet"
        frame, predictions = result_queue.queue[-1]  # show last result
        img_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return img_display, str(predictions)

    with gr.Blocks() as demo:
        gr.Markdown("### Realtime Defect Detection Monitor")
        img = gr.Image(label="Latest Frame")
        result = gr.Textbox(label="Predictions")
        refresh_btn = gr.Button("Refresh Latest Result")

        refresh_btn.click(fn=display_latest_result, outputs=[img, result])

    demo.launch(share=False)

def main():
    cam_thread = threading.Thread(target=camera_capture, args=(0,), daemon=True)
    detect_thread = threading.Thread(target=defect_processing_thread, daemon=True)

    cam_thread.start()
    detect_thread.start()

    logging.info("\U0001F4E1 System ready. Waiting for PLC trigger and inference results.")
    gradio_interface()

if __name__ == "__main__":
    main()
