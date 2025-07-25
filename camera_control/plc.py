import signal
import threading
import time
import snap7
import logging
import queue
from queue import Queue
import cv2
from datetime import datetime
import gradio as gr
import snap7
from snap7.util import get_bool, set_bool

class CameraThread(threading.Thread):

    def __init__(self, log_queue, trigger_event, stop_event):
        super().__init__()
        self.log_queue = log_queue  # 用于传递日志到主线程
        self.trigger_event = trigger_event  # PLC触发事件
        self.stop_event = stop_event  # 停止线程事件
        self.camera = None             # 相机管理器
        self.last_capture_time = None
        self.plc = None                     # PLC 客户端
        self.plc_ip = "192.168.0.1"         # PLC IP 地址
        self.plc_db = 1                     # PLC 数据块编号
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

    def connect_plc(self):
        """连接 PLC"""
        try:
            self.plc = snap7.client.Client()
            self.plc.connect(self.plc_ip, 0, 1)  # 参数: IP, 机架号, 槽号
            self.logger.info("PLC 连接成功")
            return True
        except Exception as e:
            self.logger.error(f"PLC 连接失败: {e}")
            return False
        
    def read_plc_trigger(self):
        """读取 PLC 触发信号 (DB1.DBX0.0)"""
        try:
            data = self.plc.db_read(self.plc_db, 0, 1)  # 读取 DB1.DBB0
            trigger_bit = get_bool(data, 0, 0)          # 解析 DB1.DBX0.0
            return trigger_bit
        except Exception as e:
            self.logger.error(f"读取 PLC 触发信号失败: {e}")
            return False    

    def reset_plc_trigger(self):
        """复位 PLC 触发信号 (DB1.DBX0.0)"""
        try:
            data = self.plc.db_read(self.plc_db, 0, 1)
            set_bool(data, 0, 0, False)  # 将 DB1.DBX0.0 设为 False
            self.plc.db_write(self.plc_db, 0, data)
        except Exception as e:
            self.logger.error(f"复位 PLC 触发信号失败: {e}")

    def write_plc_results(self, results):
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

    def run(self):
        self.logger.info("准备启动PLC信号监测")
        try:
            # 1. 初始化相机  
            # self.camera = CameraManager() 
            # self.logger.info("获取相机实例")
            # if self.camera.data_camera() is None:
            #     self.logger.error("相机初始化失败")
            #     return
            # else:
            #     self.logger.info(f"当前相机链接状态:{self.camera.device_status}")
            # 2. 连接 PLC
            if not self.connect_plc():
                self.logger.info("PLC连接失败")
                return
            # 3. 主循环：监测 PLC 触发信号
            while not self.stop_event.is_set():
                try:
                    # 检查 PLC 触发信号
                    if self.read_plc_trigger():
                        self.logger.info("检测到 PLC 触发信号")
                        self.reset_plc_trigger()  # 复位触发位
                        self.camera.capture_and_save()
                    time.sleep(0.1)  # 100ms 轮询间隔
                except Exception as e:
                    self.logger.error(f"主循环异常: {e}")
                    time.sleep(1)             
        except Exception as e:
            self.logger.error(f"相机线程异常: {str(e)}")
        finally:
            if self.camera and self.camera.device_status:
                self.camera.off_camera()
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

class PLCIntegration:   # 不用这个类
    def __init__(self, trigger_event):
        self.plc = snap7.client.Client()
        self.trigger_event = trigger_event
        self.connected = False
        
    def connect(self, ip):
        try:
            self.plc.connect(ip, 0, 1)
            self.connected = True
            return True
        except Exception as e:
            print(f"PLC连接失败: {e}")
            return False
            
    def monitor_trigger(self):
        while self.connected:
            try:
                # 读取PLC触发信号 (DB1.DBX0.0)
                data = self.plc.db_read(1, 0, 1)
                if data[0] & 0x01:  # 检查最低位
                    self.trigger_event.set()  # 触发相机线程
                    # 确认接收 (复位PLC触发位)
                    self.plc.db_write(1, 0, bytearray([0]))
            except Exception as e:
                print(f"PLC通信错误: {e}")
                time.sleep(1)
                
    def send_results(self, results):
        if self.connected:
            # 将结果写入PLC (示例: DB1.DBB4-DBB6)
            result_bytes = bytearray([
                1 if results['position1'] == 'OK' else 0,
                1 if results['position2'] == 'OK' else 0,
                1 if results['position3'] == 'OK' else 0
            ])
            self.plc.db_write(1, 4, result_bytes)


def main():
    # ✅ 主线程初始化 logging（影响整个进程）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]  # 输出到控制台
    )
    # 创建线程间通信组件
    log_queue = Queue()
    trigger_event = threading.Event()
    stop_event = threading.Event()
    
    # 创建相机线程
    camera_thread = CameraThread(log_queue, trigger_event, stop_event, name="CameraThread")
    
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