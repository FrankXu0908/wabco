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
from siemens_s7_1200_client import SiemensS71200Client
from detector import ObjectDetector, DefectClassifier

class CameraThread(threading.Thread):

    def __init__(self, trigger_event, stop_event):
        super().__init__()
        self.log_queue = Queue()  # 用于传递日志到主线程
        self.image_queue = Queue()  # 用于传递图像数据
        self.trigger_event = trigger_event  # PLC触发事件
        self.stop_event = stop_event  # 停止线程事件
        self.camera = None             # 相机管理器
        self.last_capture_time = None
        self.plc = None                     # PLC 客户端
        self.plc_ip = "192.168.3.100"      # PLC的IP地址
        self.plc_db = 79                     # PLC 数据块编号
        self.plc_rack = 0  # PLC的机架号
        self.plc_slot = 1  # PLC的槽位号
        self.setup_logger()


    def setup_logger(self):
        """设置线程专用日志器"""
        self.logger = logging.getLogger('camera_thread')
        self.logger.setLevel(logging.INFO)
        
        # 自定义日志处理器，将日志放入队列
        queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        queue_handler.setFormatter(formatter)
        self.logger.addHandler(queue_handler)
        """将拍照结果写入 PLC (DB1.DBB4-DBB6)"""
        try:
            # 将结果转换为字节 (示例: OK=1, NG=0)
            result_bytes = bytearray([
                1 if results['position1'] == 'OK' else 0,
                1 if results['position2'] == 'OK' else 0,
                1 if results['position3'] == 'OK' else 0
            ])
            self.plc.db_write(self.plc_db, 4, result_bytes)  # 写入 DB1.DBB4
            self.logger.info("结果已写入 PLC")
        except Exception as e:
            self.logger.error(f"写入 PLC 结果失败: {e}")

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

    def send_results(self, preds):
        # 将结果写入PLC (示例: DB1.DBB4-DBB6)
        result_bytes = bytearray([preds[0], preds[1], preds[2], preds[3], preds[4], preds[5]])
        self.plc.write_data_block(self.plc_db, 4, result_bytes)

    def run(self):
        self.logger.info("准备启动相机线程")
        try:
            # 1. 获取相机实例并且初始化
            self.camera = MultiCameraManager() 
            self.logger.info("获取相机实例")
            self.camera.initialize_all_cameras()
            # 2. 连接 PLC并开始监测，返回画框到照片队列
            self.plc = SiemensS71200Client(self.plc_ip, self.plc_rack, self.plc_slot)
            if not self.plc.connect_to_plc():
                self.logger.info("PLC连接失败")
                return
            self.plc.register_callback(self.handle_trigger)
            monitor_and_capture_thread = threading.Thread(target=self.plc.start_monitoring, kwargs={'interval': 0.3})
            monitor_and_capture_thread.start()
            # Main loop to process frames
            self.classifier = DefectClassifier()  # 初始化分类器
            while not self.stop_event.is_set():
                try:
                    camera_id, frames = self.image_queue.get(timeout=0.5)
                    if frames is not None:
                        results, preds = self.classifier.classify(frames, camera_id)
                        self.logger.info(f"相机 {camera_id} 识别结果: {results}")
                        self.send_results(preds)
                except Empty:
                    continue
                time.sleep(0.5)
            self.plc.stop_monitoring()
            monitor_and_capture_thread.join(timeout=2.0)
        except Exception as e:
            self.logger.error(f"相机线程异常: {str(e)}")
        finally:
            if self.camera:
                self.camera.close_all_cameras()
            self.logger.info("相机线程停止")

def create_gradio_interface(log_queue):
    with gr.Blocks(title="PLC相机控制系统") as demo:
        # 状态显示区域
        with gr.Row():
            camera_status = gr.Textbox(label="相机状态", interactive=False)
            last_capture = gr.Textbox(label="最后拍摄时间", interactive=False)
            results_display = gr.JSON(label="识别结果")
        
        # 日志显示区域
        log_output = gr.Textbox(label="系统日志", lines=15, interactive=False)
        
        # 控制按钮
        with gr.Row():
            start_btn = gr.Button("启动系统")
            stop_btn = gr.Button("停止系统")
            manual_trigger = gr.Button("手动触发")
            
        # 实时更新日志的函数
        def update_log():
            while True:
                if not log_queue.empty():
                    yield log_queue.get()
                else:
                    yield None
                    time.sleep(0.1)
        
        # 组件事件绑定
        demo.load(
            fn=lambda: {"相机状态": "待机", "最后拍摄时间": "未拍摄"},
            outputs=[camera_status, last_capture]
        )
        
        log_output.change(
            fn=update_log,
            inputs=[],
            outputs=log_output,
            every=0.1
        )
        
    return demo

class QueueHandler(logging.Handler):
    """自定义日志处理器，将日志放入队列"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        self.log_queue.put(self.format(record))


def main():
    # ✅ 主线程初始化 logging（影响整个进程）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]  # 输出到控制台
    )
    # 创建线程间通信组件
    trigger_event = threading.Event()
    stop_event = threading.Event()
    
    # 创建相机线程
    camera_thread = CameraThread(trigger_event, stop_event)
    
    # 创建Gradio界面
    #demo = create_gradio_interface(log_queue)
    
    # 启动线程
    camera_thread.start()
    
    # 启动Gradio界面
    #demo.launch()
    
    #主线程
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        camera_thread.join(timeout=2.0)
        if camera_thread.is_alive():
            logging.warning("警告: 相机线程未及时停止！")
        logging.info("主线程退出")


if __name__ == "__main__":
    main()