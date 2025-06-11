#!/usr/bin/env python3
"""
GPU诊断脚本 - 检查CUDA和JAX GPU支持
"""

import os
import sys
import subprocess

def run_command(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("🔍 检查NVIDIA驱动...")
    stdout, stderr, code = run_command("nvidia-smi")
    
    if code == 0:
        print("✅ NVIDIA驱动正常")
        lines = stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"   驱动版本: {line.split('Driver Version: ')[1].split(' ')[0]}")
                break
        return True
    else:
        print("❌ NVIDIA驱动问题")
        print(f"   错误: {stderr}")
        return False

def check_cuda():
    """检查CUDA安装"""
    print("\n🔍 检查CUDA...")
    stdout, stderr, code = run_command("nvcc --version")
    
    if code == 0:
        print("✅ CUDA已安装")
        lines = stdout.split('\n')
        for line in lines:
            if 'release' in line:
                print(f"   CUDA版本: {line.split('release ')[1].split(',')[0]}")
                break
        
        # 检查CUDA路径
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        if os.path.exists(cuda_home):
            print(f"   CUDA路径: {cuda_home}")
        else:
            print("⚠️ CUDA_HOME环境变量未设置或路径不存在")
        
        return True
    else:
        print("❌ CUDA未安装或未在PATH中")
        print(f"   错误: {stderr}")
        return False

def check_jax():
    """检查JAX安装"""
    print("\n🔍 检查JAX...")
    
    try:
        import jax
        print(f"✅ JAX已安装: {jax.__version__}")
        
        # 检查设备
        devices = jax.devices()
        print(f"   总设备数: {len(devices)}")
        
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
        cpu_devices = [d for d in devices if 'cpu' in str(d).lower()]
        
        print(f"   GPU设备数: {len(gpu_devices)}")
        print(f"   CPU设备数: {len(cpu_devices)}")
        
        if gpu_devices:
            print("✅ JAX检测到GPU设备:")
            for i, device in enumerate(gpu_devices):
                print(f"     GPU {i}: {device}")
                
            # 测试GPU计算
            import jax.numpy as jnp
            test_array = jnp.array([1.0, 2.0, 3.0, 4.0])
            result = jnp.sum(test_array)
            print(f"   GPU计算测试: {result}")
            
            return True
        else:
            print("❌ JAX未检测到GPU设备")
            return False
            
    except ImportError as e:
        print(f"❌ JAX导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ JAX测试失败: {e}")
        return False

def check_environment():
    """检查环境变量"""
    print("\n🔍 检查环境变量...")
    
    important_vars = [
        'CUDA_HOME',
        'CUDA_VISIBLE_DEVICES',
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'XLA_FLAGS'
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"⚠️ {var}=未设置")

def suggest_fixes():
    """提供修复建议"""
    print("\n💡 修复建议:")
    print("1. 如果NVIDIA驱动有问题:")
    print("   sudo ubuntu-drivers autoinstall")
    print("   sudo reboot")
    print()
    print("2. 如果CUDA未安装:")
    print("   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run")
    print("   sudo sh cuda_11.8.0_520.61.05_linux.run")
    print()
    print("3. 如果JAX未检测到GPU:")
    print("   pip uninstall jax jaxlib -y")
    print("   pip install jax[cuda11_pip]==0.6.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    print()
    print("4. 设置环境变量:")
    print("   export CUDA_VISIBLE_DEVICES=0,1,2,3")
    print("   export XLA_PYTHON_CLIENT_PREALLOCATE=false")
    print("   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8")

def main():
    """主函数"""
    print("🚀 GPU诊断脚本")
    print("=" * 50)
    
    results = []
    results.append(check_nvidia_driver())
    results.append(check_cuda())
    results.append(check_jax())
    
    check_environment()
    
    print("\n" + "=" * 50)
    print("📋 诊断结果:")
    
    if all(results):
        print("✅ 所有检查通过！GPU环境配置正确。")
    else:
        print("❌ 发现问题，请参考修复建议。")
        suggest_fixes()

if __name__ == "__main__":
    main()
