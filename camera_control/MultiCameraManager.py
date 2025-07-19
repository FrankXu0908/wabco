import logging
from MvCameraControl_class import *
from CameraParams_header import *
from ctypes import *
import time
import cv2
import numpy as np
import os
from datetime import datetime
import signal
import sys
from siemens_s7_1200_client import SiemensS71200Client

class MultiCameraManager:
    def __init__(self):
        self.cameras = {}  # 存储多个相机实例
        self.data_bufs = {}  # 每个相机的数据缓存
        self.stOutFrames = {}  # 每个相机的输出帧
        self.running = True
        # 相机配置
        self.config = {
                0: {
                    "name": "Camera 1",
                    "channels": 0,
                    "camera_sn": "DA6426319",
                    "params": {
                        "exposure": 2000.0,
                        "gain_value": 0,
                        "offset_y": 0,
                        "process": "left",
                        "width": 2448,
                        "height":2048,  
                        }
                    },
                1: {
                    "name": "Camera 2",
                    "channels": 1 ,
                    "camera_sn": "DA6426332",
                    "params": {
                            "exposure": 2000.0,
                            "gain_value": 0,
                            "offset_y": 0,
                            "process": "right",  
                            "width": 2448,
                            "height":2048,  
                    }
                }
            }
        self.logger = logging.getLogger("MultiCameraManager")
        self.last_signal_states = {}
        for cam_id, cam_config in self.config.items():
            for line_id in cam_config["channels"].keys():
                self.last_signal_states[f"{cam_id}_{line_id}"] = False
        
        self.signal_callbacks = []

    def initialize_camera(self, camera_id, camera_config):
        """初始化单个相机"""
        # 创建相机实例
        cam = MvCamera()
        
        # 枚举设备
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = cam.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print(f"枚举设备失败! ret=[0x{ret:x}]")
            return None

        if deviceList.nDeviceNum == 0:
            print("没有找到设备!")
            return None

        # 查找指定序列号的相机
        target_serial = camera_config["camera_sn"]
        found_device = False

        for i in range(deviceList.nDeviceNum):
            stDeviceList = cast(deviceList.pDeviceInfo[int(i)], POINTER(MV_CC_DEVICE_INFO)).contents
            temp_cam = MvCamera()
            ret = temp_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                continue

            ret = temp_cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
            if ret != 0:
                temp_cam.MV_CC_DestroyHandle()
                continue

            stStringValue = MVCC_STRINGVALUE()
            ret = temp_cam.MV_CC_GetStringValue("DeviceSerialNumber", stStringValue)
            if ret == 0:
                serial_number = stStringValue.chCurValue.decode('ascii')
                if serial_number == target_serial:
                    found_device = True
                    temp_cam.MV_CC_CloseDevice()
                    temp_cam.MV_CC_DestroyHandle()
                    ret = cam.MV_CC_CreateHandle(stDeviceList)
                    break

            temp_cam.MV_CC_CloseDevice()
            temp_cam.MV_CC_DestroyHandle()

        if not found_device:
            print(f"未找到序列号为 {target_serial} 的相机!")
            return None

        # 打开相机
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
        if ret != 0:
            ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print(f"打开设备失败! ret=[0x{ret:x}]")
                cam.MV_CC_DestroyHandle()
                return None

        # 设置相机参数
        self.configure_camera(cam, camera_config)

        # 开始取流
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始取流失败! ret=[0x{ret:x}]")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return None
        
        return cam

    def configure_camera(self, cam, config):
        """配置相机参数"""
        # 设置触发模式
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0:
            print(f"设置触发模式失败! ret=[0x{ret:x}]")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return None
        print("设置连续模式成功")
        # 设置触发源
        ret = cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        if ret != 0:
            print(f"设置触发源失败! ret=[0x{ret:x}]")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return None
        print("设置触发源成功")
        
        # 配置输出通道（Line1）为触发模式
        ret = cam.MV_CC_SetEnumValue("LineSelector", 1)  # 选择Line1 1 = "Line1"
        if ret != 0:
            print(f"设置LineSelector失败! ret=[0x{ret:x}]")
        else:
            print("设置LineSelector成功")

        ret = cam.MV_CC_SetEnumValue("LineMode", 8 )  # 设为Strboe模式 8 = "Strobe"
        if ret != 0:
            print(f"设置LineMode失败! ret=[0x{ret:x}]")
        else:
            print("设置LineMode成功")
        ret = cam.MV_CC_SetEnumValue("LineSource", 0)  # 曝光开始时输出信号 0 ="ExposureStart"
        if ret != 0:
            print(f"设置LineSource失败! ret=[0x{ret:x}]")
        else:
            print("设置LineSource成功")

        # 关闭自动曝光
        ret = cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        if ret != 0:
            print(f"设置自动曝光模式失败! ret=[0x{ret:x}]")
        else:
            print("关闭自动曝光成功")

        # 设置曝光时间
        exposure_time = config["params"]["exposure"]
        ret = cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret != 0:
            print(f"设置曝光时间失败! ret=[0x{ret:x}]")
        else:
            print(f"设置曝光时间为 {exposure_time} 微秒")

        # 设置增益
        gain_value = config["params"]["gain_value"]  # 适当增加增益
        ret = cam.MV_CC_SetFloatValue("Gain", gain_value)
        if ret != 0:
            print(f"设置增益失败! ret=[0x{ret:x}]")
        else:
            print(f"设置增益为 {gain_value}")
        
        # 设置图像格式
        ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        if ret != 0:
            print(f"设置图像格式失败! ret=[0x{ret:x}]")
        else:
            print("设置图像格式成功")
        
        # 设置图像分辨率和偏移
        ret = cam.MV_CC_SetIntValue("Width", config["width"])
        if ret != 0:
            print(f"设置图像宽度失败! ret=[0x{ret:x}]")
        else:
            print("设置图像宽度成功")
        ret = cam.MV_CC_SetIntValue("Height", config["height"])
        if ret != 0:
            print(f"设置图像高度失败! ret=[0x{ret:x}]")
        else:
            print("设置图像高度成功")
        ret = cam.MV_CC_SetIntValue("OffsetY", config["offset_y"])
        if ret != 0:
            print(f"设置Y偏移失败! ret=[0x{ret:x}]")
        else:
            print("设置Y偏移成功")
        
        # 设置传输参数
        ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        if ret != 0:
            print(f"设置数据包大小失败! ret=[0x{ret:x}]")
        else:
            print("设置数据包大小成功")

        ret = cam.MV_CC_SetIntValue("GevSCPD", 25000) # 从50000减少到25000
        if ret != 0:
            print(f"设置数据包间隔失败! ret=[0x{ret:x}]")
        else:
            print("设置数据包间隔成功")

    def initialize_all_cameras(self):
        """初始化所有相机"""
        for camera_id, config in self.config.items():
            cam = self.initialize_camera(camera_id, config)
            if cam is not None:
                self.cameras[camera_id] = cam
                logging.info(f"相机 {camera_id} ({config['camera_sn']}) 初始化成功")
            else:
                logging.error(f"相机 {camera_id} ({config['camera_sn']}) 初始化失败")

    def get_image(self, camera_id):
        """获取指定相机的图像"""
        cam = self.cameras.get(camera_id)
        if cam is None:
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 发送软触发命令
                time.sleep(0.2)
                ret = cam.MV_CC_SetCommandValue("TriggerSoftware")
                if ret != 0:
                    continue

                time.sleep(0.3)

                # 获取图像
                self.stOutFrames[camera_id] = MV_FRAME_OUT()
                ret = cam.MV_CC_GetImageBuffer(self.stOutFrames[camera_id], 10000)
                if ret != 0:
                    continue

                # 复制图像数据
                nPayloadSize = self.stOutFrames[camera_id].stFrameInfo.nFrameLen
                pData = self.stOutFrames[camera_id].pBufAddr
                data_buf = (c_ubyte * nPayloadSize)()
                cdll.msvcrt.memcpy(byref(data_buf), pData, nPayloadSize)

                # 释放缓存
                cam.MV_CC_FreeImageBuffer(self.stOutFrames[camera_id])
                
                return data_buf
            except Exception as e:
                print(f"获取图像时发生异常: {str(e)}")
                time.sleep(0.3)

        return None

    def capture_and_save(self, camera_id, save_path, need_split=False):
        """拍照并保存"""
        try:
            time.sleep(0.3)
            data_buf = self.get_image(camera_id)
            
            if data_buf is not None:
                try:
                    temp = np.frombuffer(data_buf, dtype=np.uint8)
                    if len(temp) > 0:
                        width = self.ni_channels[camera_id]["width"]
                        height = self.ni_channels[camera_id]["height"]
                        temp = temp.reshape((height, width))

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        if need_split:
                            # 计算中心点位置
                            center_x = width // 2
                            center_y = height // 2

                            # 获取左上角和右下角的图像
                            left_top = temp[0:center_y, 0:center_x]
                            right_bottom = temp[center_y:height, center_x:width]

                            # 保存左上角图像
                            save_path_1 = save_path + "_1"
                            os.makedirs(save_path_1, exist_ok=True)
                            image_path_1 = os.path.join(save_path_1, f"image_{timestamp}_1.bmp")
                            cv2.imwrite(image_path_1, left_top)
                            print(f"左上角图片已保存到: {image_path_1}")

                            # 保存右下角图像
                            save_path_2 = save_path + "_2"
                            os.makedirs(save_path_2, exist_ok=True)
                            image_path_2 = os.path.join(save_path_2, f"image_{timestamp}_2.bmp")
                            cv2.imwrite(image_path_2, right_bottom)
                            print(f"右下角图片已保存到: {image_path_2}")
                        else:
                            # 直接保存完整图片
                            os.makedirs(save_path, exist_ok=True)
                            image_path = os.path.join(save_path, f"image_{timestamp}.bmp")
                            cv2.imwrite(image_path, temp)
                            print(f"图片已保存到: {image_path}")

                        # 通知回调函数
                        for callback in self.signal_callbacks:
                            try:
                                callback()
                            except Exception as e:
                                print(f"执行回调函数时出错: {str(e)}")
                    else:
                        print("图像数据为空")
                except Exception as e:
                    print(f"处理和保存图像时发生错误: {str(e)}")
        except Exception as e:
            print(f"拍照保存时发生错误: {str(e)}")

    # def monitor_and_capture(self):
    #     """监控信号并触发拍照"""
    #     while self.running:
    #         try: 
                    
    #                 # 检测下降沿
    #                 if self.last_signal_states[f"{camera_id}_{line_id}"] and not current_state:
    #                     print(f"检测到{camera_id}_{line_id}下降沿触发信号，开始拍照...")
    #                     config = self.ni_channels[camera_id]["params"][line_id]
    #                     self.capture_and_save(
    #                         camera_id,
    #                         config["save_path"],
    #                         config["process"] == "split_quarters"
    #                     )

    #                 self.last_signal_states[f"{camera_id}_{line_id}"] = current_state
    #             time.sleep(0.001)
    #             continue


    def signal_handler(self, signum, frame):
        """处理退出信号"""
        print("\n正在安全退出程序...")
        self.running = False
        self.close_all_cameras()
        sys.exit(0)

    def close_all_cameras(self):
        """关闭所有相机"""
        for camera_id, cam in self.cameras.items():
            try:
                cam.MV_CC_StopGrabbing()
                cam.MV_CC_CloseDevice()
                cam.MV_CC_DestroyHandle()
                print(f"相机 {camera_id} 已关闭")
            except Exception as e:
                print(f"关闭相机 {camera_id} 时发生错误: {str(e)}")

    def add_signal_callback(self, callback):
        """添加信号回调函数"""
        self.signal_callbacks.append(callback)

    def remove_signal_callback(self, callback):
        """移除信号回调函数"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)

    def capture_and_return(self,camera_id):
        """拍照并返回"""
        try:
            time.sleep(0.1) 
            # 尝试获取图像
            self.data_bufs[camera_id] = self.get_image(camera_id)
            if self.data_bufs[camera_id] is not None:
                try:
                    # 处理图像数据
                    temp = np.frombuffer(self.data_bufs[camera_id], dtype=np.uint8)
                    if len(temp) > 0:
                        width = self.stOutFrames[camera_id].stFrameInfo.nWidth   # 2448
                        height = self.stOutFrames[camera_id].stFrameInfo.nHeight # 2048
                        temp = temp.reshape((height, width))

                    #     # 计算中心点
                    #     center_y = height // 2  # 1024
                    #     # 只保留下半部分（从中心点到底部）
                    #     temp = temp[center_y:, :]  # 修改这里，保留所有列（宽度），只取下半部分高度
                        # 返回处理后的图像数据
                        return temp

                except Exception as e:
                    print(f"处理和保存图像时发生错误: {str(e)}")

        except Exception as e:
            print(f"拍照过程中发生错误: {str(e)}")
            
def main():
    manager = MultiCameraManager()
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, manager.signal_handler)
    
    # 初始化所有相机
    manager.initialize_all_cameras()
    
    # 开始监控和拍照
    manager.monitor_and_capture()


if __name__ == "__main__":
    main() 