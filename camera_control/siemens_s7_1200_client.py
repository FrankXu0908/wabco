#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西门子PLC S7-1200 (1212C DC/DC/RLY) 通信客户端

本程序用于建立与西门子PLC S7-1200系列的TCP/IP连接，
支持数据读写、状态监控和批量操作功能。

PLC型号: S7-1212C DC/DC/RLY
- CPU: S7-1212C
- 数字输入: 8点 DC 24V
- 数字输出: 6点 继电器输出 (RLY)
- 模拟输入: 2路
- 通信接口: 以太网端口

作者: Robot Control System
创建日期: 2024
"""

from queue import Queue
import time
import logging
import threading
from typing import Dict, List, Optional, Union, Any
from datetime import datetime  #2025-06-10修改
import json
import snap7
from snap7.util import *
from snap7 import Area
# try:
#     import snap7
#     from snap7.util import *
#     from snap7 import Areas
# except ImportError:
#     print("错误: 未安装python-snap7库")
#     print("请运行: pip install python-snap7")
#     print("注意: 还需要安装Snap7库的C++组件")
#     exit(1)


class SiemensS71200Client:
    """
    西门子S7-1200 PLC通信客户端
    
    支持的功能:
    - TCP/IP连接管理
    - 数据块(DB)读写
    - 输入/输出状态读取
    - 标记(Mark)内存访问
    - 批量数据操作
    - 实时监控
    """
    
    def __init__(self, plc_ip_address: str, plc_rack: int = 0, plc_slot: int = 1, 
                 connection_timeout: int = 5):
        """
        初始化PLC客户端
        
        参数:
            plc_ip_address: PLC的IP地址
            plc_rack: PLC机架号 (S7-1200通常为0)
            plc_slot: PLC插槽号 (S7-1200通常为1)
            connection_timeout: 连接超时时间(秒)
        """
        self.plc_ip_address = plc_ip_address
        self.plc_rack = plc_rack
        self.plc_slot = plc_slot
        self.connection_timeout = connection_timeout
        self.plc_db = 79
        self.running = False
        self.callback = None
        
        # 初始化Snap7客户端
        self.plc_client = snap7.client.Client()
        
        # 连接状态标志
        self.is_connected = False
        self.connection_lock = threading.Lock()
        
        # 设置日志记录
        self.setup_logging()
        
        # 数据缓存
        self.data_cache = {}
        self.last_update_time = None
        
        self.logger.info(f"西门子S7-1200客户端初始化完成")
        self.logger.info(f"目标PLC: {plc_ip_address}, 机架: {plc_rack}, 插槽: {plc_slot}")

    def setup_logging(self):
        """设置日志记录配置"""
        self.logger = logging.getLogger('SiemensS71200')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('siemens_s7_1200.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def connect_to_plc(self) -> bool:
        """
        连接到PLC
        
        返回:
            bool: 连接成功返回True，否则返回False
        """
        with self.connection_lock:
            try:
                self.logger.info(f"正在连接到PLC: {self.plc_ip_address}")
                
                # 设置连接超时（兼容不同版本的python-snap7）
                self._set_timeout_safely()
                
                # 建立连接
                self.plc_client.connect(self.plc_ip_address, self.plc_rack, self.plc_slot)
                
                # 验证连接状态（兼容不同版本）
                if self._get_connection_status():
                    self.is_connected = True
                    self.logger.info("PLC连接成功!")
                    
                    # 获取PLC信息
                    plc_info = self.get_plc_info()
                    if plc_info:
                        self.logger.info(f"PLC信息: {plc_info}")
                    
                    return True
                else:
                    self.logger.error("PLC连接失败: 连接状态验证失败")
                    return False
                    
            except Exception as e:
                self.logger.error(f"连接PLC时发生错误: {str(e)}")
                self.is_connected = False
                return False

    def _set_timeout_safely(self):
        """安全设置连接超时，兼容不同版本的python-snap7"""
        timeout_methods = [
            'set_connection_timeout',
            'set_recv_timeout', 
            'set_send_timeout',
            'set_timeout'
        ]
        
        for method_name in timeout_methods:
            if hasattr(self.plc_client, method_name):
                try:
                    method = getattr(self.plc_client, method_name)
                    method(self.connection_timeout)
                    self.logger.debug(f"成功设置超时: {method_name}({self.connection_timeout})")
                    return
                except Exception as e:
                    self.logger.debug(f"设置超时失败 {method_name}: {str(e)}")
                    continue
        
        self.logger.warning("未能设置连接超时，将使用默认值")

    def _get_connection_status(self) -> bool:
        """安全获取连接状态，兼容不同版本的python-snap7"""
        status_methods = [
            'get_connected',
            'get_connection_status', 
            'connected'
        ]
        
        for method_name in status_methods:
            if hasattr(self.plc_client, method_name):
                try:
                    method = getattr(self.plc_client, method_name)
                    status = method()
                    # 处理不同的返回值类型
                    if isinstance(status, bool):
                        return status
                    elif isinstance(status, int):
                        return status == 0  # 在某些版本中，0表示连接成功
                    else:
                        return bool(status)
                except Exception as e:
                    self.logger.debug(f"获取连接状态失败 {method_name}: {str(e)}")
                    continue
        
        # 如果没有可用的状态检查方法，尝试简单的操作来验证连接
        try:
            # 尝试读取一个小的数据块来验证连接
            test_data = self.plc_client.db_read(1, 0, 1)
            return True
        except:
            return False

    def disconnect_from_plc(self):
        """断开PLC连接"""
        with self.connection_lock:
            try:
                if self.is_connected:
                    self.plc_client.disconnect()
                    self.is_connected = False
                    self.logger.info("已断开PLC连接")
            except Exception as e:
                self.logger.error(f"断开连接时发生错误: {str(e)}")

    def get_plc_info(self) -> Optional[Dict[str, Any]]:
        """
        获取PLC基本信息
        
        返回:
            dict: PLC信息字典，如果失败返回None
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法获取信息")
                return None
                
            # 尝试获取CPU信息（某些PLC可能不支持或有权限限制）
            try:
                # 重定向stderr来抑制snap7库的错误输出
                import sys
                import os
                from contextlib import redirect_stderr
                
                # 获取空设备路径（Windows: 'nul', Unix: '/dev/null'）
                null_device = 'nul' if os.name == 'nt' else '/dev/null'
                
                with redirect_stderr(open(null_device, 'w')):
                    cpu_info = self.plc_client.get_cpu_info()
                
                plc_info = {
                    "模块类型名称": cpu_info.ModuleTypeName.decode('ascii', errors='ignore'),
                    "序列号": cpu_info.SerialNumber.decode('ascii', errors='ignore'),
                    "AS名称": cpu_info.ASName.decode('ascii', errors='ignore'),
                    "版本": f"{cpu_info.ModuleName.decode('ascii', errors='ignore')}",
                    "连接时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return plc_info
            except Exception as cpu_error:
                # 如果获取CPU信息失败，返回基本连接信息
                self.logger.info("CPU详细信息不可访问，使用基本信息")
                
                plc_info = {
                    "PLC地址": self.plc_ip_address,
                    "连接状态": "已连接",
                    "连接时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "PLC型号": "S7-1212C DC/DC/RLY",
                    "说明": "PLC连接正常，基本功能可用"
                }
                return plc_info
            
        except Exception as e:
            self.logger.error(f"获取PLC信息时发生错误: {str(e)}")
            return None

    def read_data_block(self, db_number: int, start_address: int, 
                       data_size: int) -> Optional[bytearray]:
        """
        读取数据块(DB)内容
        
        参数:
            db_number: 数据块编号
            start_address: 起始地址
            data_size: 读取的字节数
            
        返回:
            bytearray: 读取的数据，如果失败返回None
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法读取数据块")
                return None
            
            # 检查数据块是否存在（通过尝试读取第一个字节）
            if start_address == 0 and data_size > 1:
                try:
                    # 先尝试读取1个字节来检查DB是否存在
                    test_data = self.plc_client.db_read(db_number, 0, 1)
                except Exception as test_error:
                    if "Address out of range" in str(test_error) or "Item not available" in str(test_error):
                        self.logger.warning(f"数据块DB{db_number}可能不存在或无访问权限")
                        return None
                    raise test_error
                
            data = self.plc_client.db_read(db_number, start_address, data_size)
            self.logger.debug(f"成功读取DB{db_number}.{start_address}, 大小: {data_size}字节")
            return data
            
        except Exception as e:
            error_msg = str(e)
            if "Address out of range" in error_msg:
                self.logger.warning(f"DB{db_number}地址超出范围或数据块不存在")
            elif "Item not available" in error_msg:
                self.logger.warning(f"DB{db_number}不可访问，可能需要在PLC中创建此数据块")
            else:
                self.logger.error(f"读取数据块DB{db_number}.{start_address}时发生错误: {error_msg}")
            return None

    def write_data_block(self, db_number: int, start_address: int, 
                        data: bytearray) -> bool:
        """
        写入数据块(DB)内容
        
        参数:
            db_number: 数据块编号
            start_address: 起始地址
            data: 要写入的数据
            
        返回:
            bool: 写入成功返回True，否则返回False
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法写入数据块")
                return False
            
            # 先检查数据块是否存在（尝试读取第一个字节）
            try:
                test_data = self.plc_client.db_read(db_number, 0, 1)
            except Exception as test_error:
                error_msg = str(test_error)
                if "Address out of range" in error_msg or "Item not available" in error_msg:
                    self.logger.warning(f"无法写入DB{db_number}: 数据块不存在或无访问权限")
                    self.logger.info(f"请在TIA Portal中创建数据块DB{db_number}，大小至少{start_address + len(data)}字节")
                    return False
                
            self.plc_client.db_write(db_number, start_address, data)
            self.logger.info(f"成功写入DB{db_number}.{start_address}, 大小: {len(data)}字节")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Address out of range" in error_msg:
                self.logger.warning(f"DB{db_number}地址超出范围，请检查数据块大小")
            elif "Item not available" in error_msg:
                self.logger.warning(f"DB{db_number}不可访问，请在PLC中创建此数据块")
            else:
                self.logger.error(f"写入数据块DB{db_number}.{start_address}时发生错误: {error_msg}")
            return False

    def read_digital_inputs(self, start_address: int = 0, count: int = 8) -> Optional[Dict[int, bool]]:
        """
        读取数字输入状态 (S7-1212C有8个数字输入)
        
        参数:
            start_address: 起始输入地址
            count: 读取的输入点数量
            
        返回:
            dict: 输入状态字典 {地址: 状态}，如果失败返回None
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法读取数字输入")
                return None
            
            # 方法1：尝试使用read_area读取输入区域（最标准的方法）
            try:
                import snap7
                # 读取输入区域 (PE区域 = Process Input)
                data = self.plc_client.read_area(snap7.types.Areas.PE, 0, start_address // 8, 1)
                if data:
                    input_byte = data[0]
                    self.logger.debug(f"使用read_area读取成功，输入字节值: {input_byte} (0x{input_byte:02X})")
                    
                    # 解析每个输入位
                    inputs = {}
                    for i in range(count):
                        bit_index = (start_address + i) % 8
                        bit_value = bool(input_byte & (1 << bit_index))
                        inputs[start_address + i] = bit_value
                    
                    self.logger.debug(f"成功读取数字输入I{start_address}.0-I{start_address + count - 1}.0")
                    return inputs
            except Exception as e1:
                self.logger.debug(f"read_area方法失败，尝试备用方法: {str(e1)}")
            
            # 方法2：备用方法 - 使用eb_read（修正参数）
            try:
                # eb_read的正确用法：第一个参数是DB号（输入用0），第二个参数是字节地址
                input_byte = self.plc_client.eb_read(0, start_address // 8)
                self.logger.debug(f"使用eb_read读取成功，输入字节值: {input_byte} (0x{input_byte:02X})")
                
                # 解析每个输入位
                inputs = {}
                for i in range(count):
                    bit_index = (start_address + i) % 8
                    bit_value = bool(input_byte & (1 << bit_index))
                    inputs[start_address + i] = bit_value
                
                self.logger.debug(f"成功读取数字输入I{start_address}.0-I{start_address + count - 1}.0")
                return inputs
            except Exception as e2:
                self.logger.debug(f"eb_read方法也失败: {str(e2)}")
            
            # 方法3：最后的备用方法 - 原始方法但修正逻辑
            try:
                input_bytes = self.plc_client.eb_read(start_address // 8, (count + 7) // 8)
                
                # 解析每个输入位
                inputs = {}
                for i in range(count):
                    byte_index = i // 8
                    bit_index = i % 8
                    if byte_index < len(input_bytes):
                        bit_value = bool(input_bytes[byte_index] & (1 << bit_index))
                        inputs[start_address + i] = bit_value
                        
                self.logger.debug(f"使用原始方法读取成功")
                return inputs
            except Exception as e3:
                self.logger.error(f"所有读取方法都失败了: 方法1({str(e1)}), 方法2({str(e2)}), 方法3({str(e3)})")
            
            return None
            
        except Exception as e:
            self.logger.error(f"读取数字输入时发生错误: {str(e)}")
            return None

    def read_digital_outputs(self, start_address: int = 0, count: int = 6) -> Optional[Dict[int, bool]]:
        """
        读取数字输出状态 (S7-1212C有6个继电器输出)
        
        参数:
            start_address: 起始输出地址
            count: 读取的输出点数量
            
        返回:
            dict: 输出状态字典 {地址: 状态}，如果失败返回None
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法读取数字输出")
                return None
                
            # 读取输出字节
            output_bytes = self.plc_client.ab_read(start_address, (count + 7) // 8)
            
            # 解析每个输出位
            outputs = {}
            for i in range(count):
                byte_index = i // 8
                bit_index = i % 8
                if byte_index < len(output_bytes):
                    bit_value = bool(output_bytes[byte_index] & (1 << bit_index))
                    outputs[start_address + i] = bit_value
                    
            self.logger.debug(f"成功读取数字输出Q{start_address}.0-Q{start_address + count - 1}.0")
            return outputs
            
        except Exception as e:
            self.logger.error(f"读取数字输出时发生错误: {str(e)}")
            return None

    def write_digital_output(self, output_address: int, bit_position: int, value: bool) -> bool:
        """
        写入单个数字输出
        
        参数:
            output_address: 输出字节地址
            bit_position: 位位置 (0-7)
            value: 输出值 (True/False)
            
        返回:
            bool: 写入成功返回True，否则返回False
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法写入数字输出")
                return False
                
            # 读取当前输出字节
            current_byte = self.plc_client.ab_read(output_address, 1)
            
            # 修改指定位
            if value:
                current_byte[0] |= (1 << bit_position)  # 设置位
            else:
                current_byte[0] &= ~(1 << bit_position)  # 清除位
                
            # 写回修改后的字节
            self.plc_client.ab_write(output_address, current_byte)
            
            self.logger.info(f"成功写入数字输出Q{output_address}.{bit_position} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"写入数字输出Q{output_address}.{bit_position}时发生错误: {str(e)}")
            return False

    def read_analog_inputs(self, start_address: int = 0, count: int = 2) -> Optional[Dict[int, float]]:
        """
        读取模拟输入值 (S7-1212C有2个模拟输入)
        
        参数:
            start_address: 起始模拟输入地址
            count: 读取的模拟输入数量
            
        返回:
            dict: 模拟输入值字典 {地址: 值}，如果失败返回None
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法读取模拟输入")
                return None
                
            # 读取模拟输入字 (每个模拟输入占2字节)
            analog_data = self.plc_client.eb_read(start_address * 2, count * 2)
            
            # 解析模拟值
            analog_values = {}
            for i in range(count):
                # 从字节数组中提取16位整数值
                word_value = (analog_data[i * 2] << 8) | analog_data[i * 2 + 1]
                
                # S7-1200模拟输入通常是0-27648对应0-10V
                voltage = (word_value / 27648.0) * 10.0
                analog_values[start_address + i] = round(voltage, 3)
                
            self.logger.debug(f"成功读取模拟输入AI{start_address}-AI{start_address + count - 1}")
            return analog_values
            
        except Exception as e:
            self.logger.error(f"读取模拟输入时发生错误: {str(e)}")
            return None

    def read_marker_memory(self, start_address: int, data_size: int) -> Optional[bytearray]:
        """
        读取标记内存(M区)数据
        
        参数:
            start_address: 起始地址
            data_size: 读取的字节数
            
        返回:
            bytearray: 读取的数据，如果失败返回None
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法读取标记内存")
                return None
                
            data = self.plc_client.mb_read(start_address, data_size)
            self.logger.debug(f"成功读取标记内存M{start_address}, 大小: {data_size}字节")
            return data
            
        except Exception as e:
            self.logger.error(f"读取标记内存M{start_address}时发生错误: {str(e)}")
            return None

    def write_marker_memory(self, start_address: int, data: bytearray) -> bool:
        """
        写入标记内存(M区)数据
        
        参数:
            start_address: 起始地址
            data: 要写入的数据
            
        返回:
            bool: 写入成功返回True，否则返回False
        """
        try:
            if not self.is_connected:
                self.logger.warning("PLC未连接，无法写入标记内存")
                return False
            
            # 检查不同版本的API签名
            try:
                # 尝试标准调用
                self.plc_client.mb_write(start_address, len(data), data)
            except TypeError:
                try:
                    # 尝试另一种参数顺序
                    self.plc_client.mb_write(start_address, data)
                except Exception as e2:
                    self.logger.error(f"标记内存写入API调用失败: {str(e2)}")
                    return False
            
            self.logger.info(f"成功写入标记内存M{start_address}, 大小: {len(data)}字节")
            return True
            
        except Exception as e:
            self.logger.error(f"写入标记内存M{start_address}时发生错误: {str(e)}")
            return False

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        获取PLC综合状态信息
        
        返回:
            dict: 包含所有状态信息的字典
        """
        status_report = {
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "PLC地址": self.plc_ip_address,
            "连接状态": self.is_connected,
            "数字输入状态": {},
            "数字输出状态": {},
            "模拟输入值": {},
            "PLC信息": {}
        }
        
        if not self.is_connected:
            status_report["错误信息"] = "PLC未连接"
            return status_report
            
        try:
            # 读取数字输入状态 (I0.0-I0.7)
            digital_inputs = self.read_digital_inputs(0, 8)
            if digital_inputs:
                for addr, value in digital_inputs.items():
                    status_report["数字输入状态"][f"I0.{addr}"] = value
                    
            # 读取数字输出状态 (Q0.0-Q0.5)
            digital_outputs = self.read_digital_outputs(0, 6)
            if digital_outputs:
                for addr, value in digital_outputs.items():
                    status_report["数字输出状态"][f"Q0.{addr}"] = value
                    
            # 读取模拟输入值 (AI0, AI1)
            analog_inputs = self.read_analog_inputs(0, 2)
            if analog_inputs:
                for addr, value in analog_inputs.items():
                    status_report["模拟输入值"][f"AI{addr}"] = f"{value}V"
                    
            # 获取PLC信息
            plc_info = self.get_plc_info()
            if plc_info:
                status_report["PLC信息"] = plc_info
                
        except Exception as e:
            status_report["错误信息"] = f"读取状态时发生错误: {str(e)}"
            
        return status_report

    def print_status_report(self, status_data: Dict[str, Any]):
        """
        打印格式化的状态报告
        
        参数:
            status_data: 状态数据字典
        """
        print("=" * 80)
        print(f"西门子PLC S7-1212C 状态报告 - {status_data.get('时间戳', '')}")
        print(f"PLC地址: {status_data.get('PLC地址', '')}")
        print("=" * 80)
        
        # 连接状态
        connection_status = "已连接" if status_data.get("连接状态", False) else "未连接"
        print(f"\n【连接状态】: {connection_status}")
        
        # PLC信息
        if "PLC信息" in status_data and status_data["PLC信息"]:
            print(f"\n【PLC信息】")
            for key, value in status_data["PLC信息"].items():
                print(f"  {key}: {value}")
        
        # 数字输入状态
        if "数字输入状态" in status_data and status_data["数字输入状态"]:
            print(f"\n【数字输入状态】")
            for input_name, value in status_data["数字输入状态"].items():
                status_text = "ON" if value else "OFF"
                print(f"  {input_name}: {status_text}")
        
        # 数字输出状态
        if "数字输出状态" in status_data and status_data["数字输出状态"]:
            print(f"\n【数字输出状态】")
            for output_name, value in status_data["数字输出状态"].items():
                status_text = "ON" if value else "OFF"
                print(f"  {output_name}: {status_text}")
        
        # 模拟输入值
        if "模拟输入值" in status_data and status_data["模拟输入值"]:
            print(f"\n【模拟输入值】")
            for analog_name, value in status_data["模拟输入值"].items():
                print(f"  {analog_name}: {value}")
        
        # 错误信息
        if "错误信息" in status_data:
            print(f"\n【错误信息】: {status_data['错误信息']}")
        
        print("=" * 80)

    def start_monitoring(self, interval: float = 2.0, max_duration: Optional[float] = None):
        """
        启动PLC状态监控
        
        参数:
            interval: 监控间隔时间(秒)
            max_duration: 最大监控时间(秒)，None表示无限制
        """
        self.logger.info(f"开始PLC监控，间隔: {interval}秒")      
        start_time = time.time()     
        self.running = True
        try:
            while self.running:
                # 获取触发位状态
                data = self.read_data_block(self.plc_db, 0, 1)
                trigger_bit_left = get_bool(data, 0, 0)
                trigger_bit_right = get_bool(data,0,1)
                trigger_bit_side = get_bool(data,0,2)
                trigger_bit = trigger_bit_left or trigger_bit_right or trigger_bit_side
                if trigger_bit:
                    trigger_info = (trigger_bit_left, trigger_bit_right, trigger_bit_side)
                    self.logger.info(f"触发位状态: {trigger_bit}, 左: {trigger_bit_left}, 右: {trigger_bit_right}, 侧面: {trigger_bit_side}，开始拍照")
                    if self.callback:
                        self.callback(trigger_info)  # invoke the callback with signal info

                # 检查是否达到最大监控时间
                if max_duration and (time.time() - start_time) > max_duration:
                    self.logger.info("达到最大监控时间，停止监控")
                    break
                
                # # 获取并显示状态
                # status = self.get_comprehensive_status()
                # self.print_status_report(status)
                
                # # 缓存数据
                # self.data_cache = status
                self.last_update_time = datetime.now()
                
                # 等待下次监控
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("用户中断监控")
        except Exception as e:
            self.logger.error(f"监控过程中发生错误: {str(e)}")

    def stop_monitoring(self):
        self.running = False
    
    def register_callback(self, callback_fn):
        self.callback = callback_fn
        self.logger.info("Callback function registered.")
        
    def __enter__(self):
        """上下文管理器入口"""
        if self.connect_to_plc():
            return self
        else:
            raise ConnectionError("无法连接到PLC")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect_from_plc()


def main():
    """
    主函数 - 演示程序使用方法
    """
    print("西门子PLC S7-1212C DC/DC/RLY 通信程序")
    print("=" * 50)
    
    # PLC连接配置
    PLC_IP_ADDRESS = "192.168.3.100"  # 请修改为您的PLC实际IP地址192.168.3.100
    PLC_RACK = 0  # S7-1200通常为0
    PLC_SLOT = 1  # S7-1200通常为1
    
    print(f"PLC IP地址: {PLC_IP_ADDRESS}")
    print(f"PLC机架: {PLC_RACK}")
    print(f"PLC插槽: {PLC_SLOT}")
    print("=" * 50)
    
    # 使用上下文管理器确保正确关闭连接
    try:
        with SiemensS71200Client(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT) as plc:
            print("PLC连接成功!")
            print("开始监控... 按 Ctrl+C 停止监控")    
            # 启动监控 (每2秒更新一次)
            plc.start_monitoring(interval=0.3)
            
    except ConnectionError as e:
        print(f"连接错误: {str(e)}")
    except Exception as e:
        print(f"程序运行错误: {str(e)}")
    
    print("程序已结束")


if __name__ == "__main__":
    main()