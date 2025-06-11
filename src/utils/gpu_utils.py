"""
GPU工具函数
"""

import os
import jax
from typing import List


def setup_jax_environment():
    """设置JAX环境配置"""
    # JAX配置优化
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    # 清理可能存在的XLA_FLAGS
    if 'XLA_FLAGS' in os.environ:
        del os.environ['XLA_FLAGS']
    
    print("🔧 JAX环境配置完成")


def check_gpu_setup():
    """检查GPU设置"""
    devices = jax.devices()
    print(f"🔍 GPU环境检查:")
    print(f"   检测到 {len(devices)} 个设备")
    
    for i, device in enumerate(devices):
        print(f"   设备 {i}: {device}")
    
    if len(devices) == 0:
        print("❌ 未检测到GPU设备，请检查CUDA和JAX安装")
        return False
    
    print(f"✅ GPU设置正常，共 {len(devices)} 个设备可用")
    return True


def get_gpu_memory_info():
    """获取GPU内存信息"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        memory_info = []
        for gpu in gpus:
            info = {
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil,
                'load': gpu.load,
                'temperature': gpu.temperature
            }
            memory_info.append(info)
        
        return memory_info
    except ImportError:
        print("⚠️ GPUtil未安装，无法获取详细GPU信息")
        return []
    except Exception as e:
        print(f"⚠️ 获取GPU信息时出错: {e}")
        return []


def print_gpu_status():
    """打印GPU状态信息"""
    memory_info = get_gpu_memory_info()
    
    if memory_info:
        print("\n💾 GPU内存状态:")
        print("-" * 50)
        
        for gpu in memory_info:
            print(f"GPU {gpu['id']} ({gpu['name']}):")
            print(f"  内存使用: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_util']*100:.1f}%)")
            print(f"  GPU利用率: {gpu['load']*100:.1f}%")
            print(f"  温度: {gpu['temperature']}°C")
            print()
    else:
        print("ℹ️ 无法获取详细GPU信息")
