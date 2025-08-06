from camera_control.MultiCameraManager import MultiCameraManager
import signal
import threading
import time
import snap7
import logging
import queue
from queue import Empty, Queue
import cv2
from datetime import datetime
import gradio as gr
import snap7
from snap7.util import get_bool, set_bool
from camera_control.siemens_s7_1200_client import SiemensS71200Client
from detector import ObjectDetector, DefectClassifier
from logging.handlers import QueueHandler, QueueListener

class CameraThread(threading.Thread):

    def __init__(self, trigger_event, stop_event):
        super().__init__()
        self.log_queue = Queue()  # 用于传递日志到主线程
        self.image_queue = Queue()  # 用于传递图像数据
        self.trigger_event = trigger_event  # PLC触发事件
        self.stop_event = stop_event  # 停止线程事件
        self.camera = None             # 相机管理器
        self.camera_status = "未初始化"  # 相机状态
        self.last_capture_time = None
        self.plc = None                     # PLC 客户端
        self.plc_ip = "192.168.3.100"      # PLC的IP地址
        self.plc_db = 79                     # PLC 数据块编号
        self.plc_rack = 0  # PLC的机架号
        self.plc_slot = 1  # PLC的槽位号
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.setup_logger()


    def setup_logger(self):
        """设置线程专用日志器"""
        self.logger = logging.getLogger('camera_thread')
        self.logger.setLevel(logging.INFO)
        
        # 自定义日志处理器，将日志放入队列
        queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        queue_handler.setFormatter(formatter)
        self.logger.addHandler(queue_handler)

    def handle_trigger(self, trigger_info):
        """Callback when PLC trigger is detected"""
        left, right, side = trigger_info
        if left:
            camera_id = 0
            frames = self.camera.capture_and_return(camera_id)
            self.image_queue.put((camera_id, frames))  # 将图像数据放入队列
        elif right:
            camera_id = 1
            frames = self.camera.capture_and_return(camera_id)
            self.image_queue.put((camera_id, frames))  # 将图像数据放入队列
        elif side:
            camera_id = 2
            frames = self.camera.capture_and_return(camera_id)
            self.image_queue.put((camera_id, frames))  # 将图像数据放入队列

    def send_results(self, camera_id, preds):
        # 将结果写入PLC (示例: DB1.DBB4-DBB6)
        result_bytes = bytearray([preds[0], preds[1], preds[2], preds[3], preds[4], preds[5]])
        if camera_id == 0:
            self.plc.write_data_block(self.plc_db, 7, result_bytes)
        if camera_id == 1:
            self.plc.write_data_block(self.plc_db, 8, result_bytes)
        if camera_id == 2:
            self.plc.write_data_block(self.plc_db, 9, result_bytes)
            

    def run(self):
        try:
            # 1. 获取相机实例并且初始化
            self.camera = MultiCameraManager() 
            self.camera.initialize_all_cameras()
            self.camera_status = "相机已初始化, 准备就绪"
            # 2. 连接 PLC并开始监测，返回画框到照片队列
            self.plc = SiemensS71200Client(self.plc_ip, self.plc_rack, self.plc_slot)
            if not self.plc.connect_to_plc():
                self.logger.info("PLC连接失败")
                return
            self.plc.register_callback(self.handle_trigger)
            monitor_and_capture_thread = threading.Thread(target=self.plc.start_monitoring, kwargs={'interval': 1}, daemon=True)
            monitor_and_capture_thread.start()
            # Main loop to process frames
            self.classifier = DefectClassifier()  # 初始化分类器
            while not self.stop_event.is_set():     
                try:
                    camera_id, frames = self.image_queue.get(timeout=0.5)
                    self.last_capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if frames is not None:
                        results, preds = self.classifier.classify(frames, camera_id)
                        self.logger.info(f"相机 {camera_id} 识别结果: {results}")
                        self.send_results(camera_id, preds)
                except Empty:
                        continue  # go back to waiting for next image
                time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"相机线程异常: {str(e)}")
        finally:
            self.plc.stop_monitoring()
            monitor_and_capture_thread.join(timeout=2.0)
            if self.camera:
                self.camera.close_all_cameras()
            self.logger.info("相机线程停止")

class GradioUI:

    def __init__(self, log_queue, ui_log_queue):
        self.log_queue = log_queue
        self.ui_log_queue = ui_log_queue
        self.camera_thread = None
        self.camera_status = "未初始化"
        self.last_capture_time = "未拍摄"
        self.results = {}
        self.ui_logs = []  # 用于存储日志信息
        self.log_lock = threading.Lock()  # <-- add this lock
        self._configure_logging()
    
    def _configure_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(name)s: %(message)s')

        # Terminal log via Queue
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(formatter)
        root_logger.addHandler(queue_handler)

        # UI log handler
        ui_log_handler = UILogHandler(self.ui_logs, self.log_lock)
        ui_log_handler.setFormatter(formatter)
        root_logger.addHandler(ui_log_handler)

        # Terminal output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.queue_listener = QueueListener(self.log_queue, stream_handler)
        self.queue_listener.start()
        
    
    def get_logs(self):
        with self.log_lock:
            return "\n".join(self.ui_logs[-100:])    
    
    def get_status(self):
        return self.camera_status
    
    def start_camera_thread(self):
        if self.camera_thread is None or not self.camera_thread.is_alive():
            # Create fresh thread and events
            self.trigger_event = threading.Event()
            self.stop_event = threading.Event()
            self.camera_thread = CameraThread(self.trigger_event, self.stop_event)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            return "相机线程已启动"
        return "相机线程已经在运行"

    def stop_camera_thread(self):
        if self.camera_thread and self.camera_thread.is_alive():
            self.stop_event.set()
            self.camera_thread.join(timeout=2.0)
            if self.camera_thread.is_alive():
                return "相机线程未能及时停止"
            self.camera_thread = None  # Allow re-creation
            return "相机线程已停止"
        return "相机线程未在运行"
            
    def create_gradio_interface(self):
        with gr.Blocks(title="PLC相机控制系统") as demo:
            # 状态显示区域
            with gr.Row():
                camera_status = gr.Textbox(label="相机状态", interactive=False)
                
            refresh_btn = gr.Button("刷新状态")
            refresh_btn.click(
                fn=self.get_status,
                inputs=[],
                outputs=[camera_status]
            )
            
            # 日志显示区域
            log_output = gr.Textbox(label="系统日志", lines=15, interactive=False)
            refresh_logs_btn = gr.Button("刷新日志")
            refresh_logs_btn.click(
                fn=self.get_logs,
                inputs=[],
                outputs=log_output
            )
            # Timer for logs
            log_timer = gr.Timer(
                value=1,  # 每秒刷新一次
                active=True,
                render=True
            )
            log_timer.tick(
                fn=self.get_logs,
                inputs=[],
                outputs=log_output
            )

            # 控制按钮
            with gr.Row():
                start_btn = gr.Button("启动系统")
                stop_btn = gr.Button("停止系统")
                manual_trigger = gr.Button("手动触发(测试中)")
                start_btn.click(fn=self.start_camera_thread, inputs=[], outputs=[camera_status])
                stop_btn.click(fn=self.stop_camera_thread, inputs=[], outputs=[camera_status])
                # manual_trigger.click(fn=self.trigger_camera_capture, inputs=[], outputs=[camera_status])

            # 组件事件绑定
            demo.load(
                    fn=self.get_status,
                    inputs=[],
                    outputs=[camera_status]
                )
            
            # demo.load(self.get_logs, inputs=[], outputs=log_output)
        
            return demo


class UILogHandler(logging.Handler):
    def __init__(self, ui_logs, log_lock):
        super().__init__()
        self.ui_logs = ui_logs
        self.log_lock = log_lock

    def emit(self, record):
        log_entry = self.format(record)
        with self.log_lock:
            self.ui_logs.append(log_entry)

def main():
    # Create the shared queue
    log_queue = Queue()
    ui_log_queue = Queue()
 
    # Initialize UI
    ui = GradioUI(log_queue, ui_log_queue)
    
    #主线程
    try:
        # 打开GUI
        demo = ui.create_gradio_interface()
        demo.launch()
        
    except KeyboardInterrupt:
        print("KeyboardInterrupt received: stopping all services.")
    finally:
        # # 停止相机线程
        # stop_event.set()
        # camera_thread.join(timeout=2.0)
        # if camera_thread.is_alive():
        #     logging.warning("警告: 相机线程未及时停止！")
        if ui.queue_listener:
            ui.queue_listener.stop()
        logging.info("主线程退出")


if __name__ == "__main__":
    main() 
    