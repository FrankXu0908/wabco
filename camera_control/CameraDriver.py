from MvCameraControl_class import *
from CameraParams_header import *
from ctypes import *
import time
import cv2
import numpy as np
import os
from datetime import datetime
import nidaqmx
import signal
import sys
from nidaqmx.constants import LineGrouping
import logging
import threading

class CameraManager:

    def __init__(self):
        self.cam = None
        self.data_buf = None
        self.device_status = False
        self.stOutFrame = None
        # self.running = True  # 控制程序运行的标志
        self.stop_event = threading.Event()
        # self.ni_device = "Dev1"  # 修改为系统建议的 Dev1
        # self.ni_channels = {
        #     "line0": {
        #         "channel": "port1/line0",
        #         "save_path": "photo",
        #         "is_capturing": False,  # 添加拍照状态标志
        #         "wait_time": 0.02  # 等待信号恢复的时间（20ms）
        #     }
        # }
        self.last_signal_states = {
            "line0": False
        }
        self.signal_callbacks = []  # 添加回调函数列表
        self.signal_changed = False

    def data_camera(self):
        # 枚举设备
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        deviceList = MV_CC_DEVICE_INFO_LIST()
        # 实例相机
        self.cam = MvCamera()
        ret = self.cam.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print(f"枚举设备失败! ret=[0x{ret:x}]")
            return None

        if deviceList.nDeviceNum == 0:
            print("没有找到设备!")
            return None

        print(f"找到 {deviceList.nDeviceNum} 个设备!")

        # 查找指定序列号的相机
        target_serial = "DA6426319" #SrpingPad第二个相机编号
        found_device = False

        for i in range(deviceList.nDeviceNum):
            stDeviceList = cast(deviceList.pDeviceInfo[int(i)], POINTER(MV_CC_DEVICE_INFO)).contents

            # 创建临时句柄来获取序列号
            temp_cam = MvCamera()
            ret = temp_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                continue

            # 打开设备
            ret = temp_cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
            if ret != 0:
                temp_cam.MV_CC_DestroyHandle()
                continue

            # 获取序列号
            stStringValue = MVCC_STRINGVALUE()
            ret = temp_cam.MV_CC_GetStringValue("DeviceSerialNumber", stStringValue)
            if ret == 0:
                serial_number = stStringValue.chCurValue.decode('ascii')
                if serial_number == target_serial:
                    found_device = True
                    # 关闭临时相机
                    temp_cam.MV_CC_CloseDevice()
                    temp_cam.MV_CC_DestroyHandle()
                    # 使用找到的设备信息创建实际的相机实例
                    ret = self.cam.MV_CC_CreateHandleWithoutLog(stDeviceList)
                    break

            # 关闭临时相机
            temp_cam.MV_CC_CloseDevice()
            temp_cam.MV_CC_DestroyHandle()

        if not found_device:
            print(f"未找到序列号为 {target_serial} 的相机!")
            return None

        # 以下是原有的相机初始化代码
        # 尝试关闭可能已经打开的设备
        self.cam.MV_CC_CloseDevice()

        # 打开相机 - 修改访问模式
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
        if ret != 0:
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print(f"打开设备失败! ret=[0x{ret:x}]")
                self.cam.MV_CC_DestroyHandle()
                return None
        print("打开相机成功")

        # ===== 相机参数配置开始 =====

        # 1. 设置为连续采集模式
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"设置触发模式失败! ret=[0x{ret:x}]")
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            return None
        print("设置连续模式成功")

        # 2. 关闭自动曝光
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        if ret != 0:
            print(f"设置自动曝光模式失败! ret=[0x{ret:x}]")
        else:
            print("关闭自动曝光成功")

        # 3. 设置曝光时间
        exposure_time = 2000.0
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret != 0:
            print(f"设置曝光时间失败! ret=[0x{ret:x}]")
        else:
            print(f"设置曝光时间为 {exposure_time} 微秒")

        # 4. 设置图像格式为Mono8
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        if ret != 0:
            print(f"设置图像格式失败! ret=[0x{ret:x}]")
        else:
            print("设置图像格式成功")

        # 5. 设置图像分辨率
        ret = self.cam.MV_CC_SetIntValue("Width", 2448)  # 设置宽度为2448
        if ret != 0:
            print(f"设置图像宽度失败! ret=[0x{ret:x}]")
        else:
            print("设置图像宽度成功")

        # 设置Y方向偏移量为0，拍摄完整图片
        ret = self.cam.MV_CC_SetIntValue("OffsetY", 0)
        if ret != 0:
            print(f"设置Y偏移失败! ret=[0x{ret:x}]")
        else:
            print("设置Y偏移成功")

        ret = self.cam.MV_CC_SetIntValue("Height", 2048)  # 设置完整高度为2048
        if ret != 0:
            print(f"设置图像高度失败! ret=[0x{ret:x}]")
        else:
            print("设置图像高度成功")

        # 6. 设置传输层参数
        ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        if ret != 0:
            print(f"设置数据包大小失败! ret=[0x{ret:x}]")
        else:
            print("设置数据包大小成功")

        # 减小数据包间隔以提高传输速度
        ret = self.cam.MV_CC_SetIntValue("GevSCPD", 25000)  # 从50000减少到25000
        if ret != 0:
            print(f"设置数据包间隔失败! ret=[0x{ret:x}]")
        else:
            print("设置数据包间隔成功")

        # 开始取流前先停止
        self.cam.MV_CC_StopGrabbing()
        time.sleep(0.1)  # 等待停止完成

        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始取流失败! ret=[0x{ret:x}]")
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            return None
        print("开始取流成功")

        self.device_status = True
        return self.cam

    def get_image(self):
        if not self.device_status or self.cam is None:
            print("相机未初始化或未连接")
            return None

        # 获取一张图像
        self.stOutFrame = MV_FRAME_OUT()
        ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 300)

        if ret != 0:
            print(f"获取图像失败! ret=[0x{ret:x}]")
            return None

        if self.stOutFrame.pBufAddr is None:
            print("图像缓冲区地址为空")
            return None

        # 获取图像数据
        nPayloadSize = self.stOutFrame.stFrameInfo.nFrameLen
        pData = self.stOutFrame.pBufAddr
        self.data_buf = (c_ubyte * nPayloadSize)()
        cdll.msvcrt.memcpy(byref(self.data_buf), pData, nPayloadSize)

        # 释放图像缓存
        self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
        return self.data_buf

    def _validate_image_data(self, data_buf, expected_size):
        """验证图像数据的有效性"""
        if data_buf is None or len(data_buf) != expected_size:
            return False

        # 将数据转换为numpy数组进行验证
        try:
            temp = np.frombuffer(data_buf, dtype=np.uint8)
            if len(temp) == 0:
                return False

            # 检查是否存在异常的像素值分布
            mean_val = np.mean(temp)
            std_val = np.std(temp)

            # 如果标准差过小或平均值异常，可能表示图像有问题
            if std_val < 5 or mean_val < 5 or mean_val > 250:
                return False

            return True
        except Exception as e:
            print(f"图像数据验证时发生错误: {str(e)}")
            return False

    def off_camera(self):

        # 停止取流
        ret = self.cam.MV_CC_StopGrabbing()
        print("停止取流执行码:[0x%x]" % ret)

        # 关闭设备
        ret = self.cam.MV_CC_CloseDevice()
        print("关闭设备执行码:[0x%x]" % ret)

        # 销毁句柄
        ret = self.cam.MV_CC_DestroyHandle()
        print("销毁句柄执行码:[0x%x]" % ret)
        self.device_status = False
        return self.device_status

    def signal_handler(self):
        """处理退出信号"""
        print("\n正在安全退出程序...")
        #self.running = False
        self.stop_event.set()
        if self.device_status:
            self.off_camera()
        sys.exit(0)

    def add_signal_callback(self, callback):
        """添加信号回调函数"""
        self.signal_callbacks.append(callback)

    def remove_signal_callback(self, callback):
        """移除信号回调函数"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)

    def get_signal_state(self):

        """获取最后一次的信号状态和是否发生变化"""
        changed = self.signal_changed
        self.signal_changed = False  # 重置变化标志
        return self.last_signal_states, changed

    def send_signal(self,output_line):
        """发送信号的具体实现"""
        try:
            message = f"准备发送信号，输出线路: {output_line}"
            logging.info(message)

            with nidaqmx.Task() as output_task:
                logging.info("创建任务成功")

                output_task.do_channels.add_do_chan(f'Dev1/port0/line{output_line}')
                logging.info("通道配置成功")

                output_task.write(False)
                logging.info("False信号写入成功")

                time.sleep(0.5)

                output_task.write(True)
                logging.info("True信号写入成功")

        except Exception as e:
            message = f"发送信号时发生错误: {str(e)}\n错误类型: {type(e)}"
            logging.error(message)

    def monitor_and_capture(self):
        try:
            # 只初始化 Port1/line0 的任务
            task = nidaqmx.Task()
            task.di_channels.add_di_chan(
                f"{self.ni_device}/port1/line0",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            task.start()

            while not self.stop_event.is_set():
                try:
                    line = "line0"
                    current_state = bool(task.read())
                    channel_config = self.ni_channels[line]  # 假设配置中保留 line0

                    # 如果当前正在拍照，等待信号恢复到高电平
                    if channel_config["is_capturing"]:
                        if current_state:  # 信号恢复到高电平
                            print(f"{line} 信号恢复到高电平，可以进行下一次触发")
                            channel_config["is_capturing"] = False
                            time.sleep(channel_config["wait_time"])  # 等待信号稳定
                        continue  # 跳过本次循环，不检测下降沿

                    # 检测下降沿（高电平变低电平）
                    if self.last_signal_states[line] and not current_state:
                        print(f"检测到{line}下降沿触发信号，开始拍照...")
                        channel_config["is_capturing"] = True  # 设置拍照状态
                        # 拍照并保存到对应文件夹
                        self.capture_and_save(channel_config["save_path"])

                    self.last_signal_states[line] = current_state
                    time.sleep(0.001)  # 1ms的采样间隔
                except Exception as e:
                    print(f"读取错误: {str(e)}")
                    time.sleep(0.1)
                    continue

            # 清理任务
            task.close()

        except Exception as e:
            print(f"任务创建错误: {str(e)}")

    def capture_and_save(self, save_path):
        try:
            # 减少等待机械动作的时间
            time.sleep(0.1)  # 从0.3减少到0.1
            # 尝试获取图像
            data_buf = self.get_image()
            if data_buf is not None:
                try:
                    # 处理图像数据
                    temp = np.frombuffer(self.data_buf, dtype=np.uint8)
                    if len(temp) > 0:
                        width = self.stOutFrame.stFrameInfo.nWidth   # 2448
                        height = self.stOutFrame.stFrameInfo.nHeight # 2048
                        temp = temp.reshape((height, width))

                    # 保存图片
                    os.makedirs(save_path, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(save_path, f"image_{timestamp}.bmp")
                    cv2.imwrite(image_path, temp)
                    # self.send_signal(output_line=0) #直接发送代表pass的信息
                    print(f"图片已保存到: {image_path}",f"保存的图片尺寸: {temp.shape}")# 应该显示 (2048, 2448)
                    return temp  # ✅ Return the captured frame

                except Exception as e:
                    print(f"处理和保存图像时发生错误: {str(e)}")

        except Exception as e:
            print(f"拍照过程中发生错误: {str(e)}")

    def check_camera_params(self):
        """检查并打印当前相机参数设置"""
        print("\n==== 当前相机参数 ====")

        # 1. 检查曝光时间
        stFloatValue = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatValue)
        if ret == 0:
            print(f"当前曝光时间: {stFloatValue.fCurValue} 微秒")
            print(f"最小曝光时间: {stFloatValue.fMin} 微秒")
            print(f"最大曝光时间: {stFloatValue.fMax} 微秒")

        # 2. 检查自动曝光状态
        stEnumValue = MVCC_ENUMVALUE()
        ret = self.cam.MV_CC_GetEnumValue("ExposureAuto", stEnumValue)
        if ret == 0:
            auto_modes = {0: "关闭", 1: "单次", 2: "连续"}
            mode = auto_modes.get(stEnumValue.nCurValue, "未知")
            print(f"自动曝光模式: {mode} ({stEnumValue.nCurValue})")

        print("=====================\n")

    def cleanup(self):
        """清理资源"""
        pass  # 不再需要清理输出任务


# 主程序
def main():
    CAM = CameraManager()

    # 可以在这里配置NI设备参数
    # CAM.ni_device = "Dev2"  # 如果需要使用不同的设备
    # CAM.ni_channel = "port0/line1"  # 如果需要使用不同的通道

    # 注册信号处理函数（用于优雅退出）
    signal.signal(signal.SIGINT, CAM.signal_handler)

    # 初始化相机
    if CAM.data_camera() is None:
        print("相机初始化失败")
        return

    print(f"当前相机链接状态:{CAM.device_status}")

    # 开始监测信号并拍照
    CAM.monitor_and_capture()


if __name__ == "__main__":
    main()
