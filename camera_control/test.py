from CameraDriver import CameraManager
import signal
import threading
import time
import snap7
import logging
import queue
import cv2
from datetime import datetime
import gradio as gr


# # 初始化
# plc = snap7.client.Client()
# plc.connect('192.168.0.1', 0, 1)  # PLC IP地址

CAM = CameraManager()
# 注册信号处理函数（用于优雅退出）
signal.signal(signal.SIGINT, CAM.signal_handler)
# 初始化相机
if CAM.data_camera() is None:
    print("相机初始化失败")

else: 
    print(f"当前相机链接状态:{CAM.device_status}")

# 启动监控线程
monitor_thread = threading.Thread(target=CAM.monitor_and_capture)
monitor_thread.daemon = True  # 主线程退出时自动终止监控线程
monitor_thread.start()
print("监控线程已启动，等待信号触发...")

# 主线程保持运行（通过无限循环或信号等待）
try:
    while True:  # 主线程持续运行
        time.sleep(1)
except KeyboardInterrupt:  # 捕获 Ctrl+C
    print("主线程退出")