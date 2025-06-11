#!/usr/bin/env python3
"""
智能JAX/CUDA检测脚本 - 准确检测CUDA支持
"""

import jax
import jax.numpy as jnp

def smart_cuda_detection():
    """智能CUDA支持检测"""
    print("🔍 智能CUDA支持检测")
    print("=" * 40)
    
    # 获取设备信息
    devices = jax.devices()
    gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or d.platform == 'gpu']
    
    print(f"JAX版本: {jax.__version__}")
    print(f"总设备数: {len(devices)}")
    print(f"GPU设备数: {len(gpu_devices)}")
    
    # CUDA支持检测
    cuda_indicators = []
    
    # 检测1: GPU设备存在
    if gpu_devices:
        cuda_indicators.append("✅ 检测到GPU设备")
        for i, device in enumerate(gpu_devices[:2]):  # 只显示前2个
            print(f"  GPU {i}: {device} ({device.device_kind})")
    else:
        cuda_indicators.append("❌ 未检测到GPU设备")
    
    # 检测2: JAXlib版本信息
    try:
        import jaxlib
        jaxlib_version = jaxlib.__version__
        print(f"\nJAXlib版本: {jaxlib_version}")
        
        if 'cuda' in jaxlib_version.lower():
            cuda_indicators.append("✅ JAXlib版本包含CUDA标识")
        else:
            cuda_indicators.append("⚠️ JAXlib版本不包含CUDA标识")
    except ImportError:
        cuda_indicators.append("❌ JAXlib未安装")
    
    # 检测3: 平台后端
    try:
        platforms = [str(d.platform) for d in devices]
        if 'gpu' in platforms:
            cuda_indicators.append("✅ JAX平台支持GPU")
        else:
            cuda_indicators.append("❌ JAX平台不支持GPU")
    except:
        cuda_indicators.append("⚠️ 无法检测平台信息")
    
    # 检测4: GPU计算测试
    if gpu_devices:
        try:
            test_array = jnp.array([1.0, 2.0, 3.0])
            with jax.default_device(gpu_devices[0]):
                result = jnp.sum(test_array)
            cuda_indicators.append(f"✅ GPU计算测试成功 (结果: {result})")
        except Exception as e:
            cuda_indicators.append(f"❌ GPU计算测试失败: {str(e)[:50]}")
    
    # 检测5: XLA后端
    try:
        from jaxlib import xla_bridge
        backend = xla_bridge.get_backend()
        backend_platform = getattr(backend, 'platform', 'unknown')
        if 'gpu' in backend_platform.lower():
            cuda_indicators.append("✅ XLA后端支持GPU")
        else:
            cuda_indicators.append(f"ℹ️ XLA后端平台: {backend_platform}")
    except Exception as e:
        cuda_indicators.append("⚠️ 无法检测XLA后端")
    
    # 综合判断
    print(f"\n🔍 CUDA支持检测结果:")
    print("-" * 30)
    
    for indicator in cuda_indicators:
        print(f"  {indicator}")
    
    # 最终判断
    gpu_count = len(gpu_devices)
    has_working_gpu = gpu_count > 0
    
    print(f"\n📊 综合评估:")
    print("-" * 20)
    
    if has_working_gpu:
        print(f"🎉 CUDA支持: ✅ 已启用")
        print(f"💪 可用GPU数量: {gpu_count}")
        print(f"🚀 推荐使用: GPU加速推理")
        
        if gpu_count >= 4:
            print(f"⚡ 多GPU并行: 支持 ({gpu_count}个GPU)")
        elif gpu_count >= 2:
            print(f"⚡ 多GPU并行: 部分支持 ({gpu_count}个GPU)")
        else:
            print(f"⚡ 多GPU并行: 不支持 (仅{gpu_count}个GPU)")
            
    else:
        print(f"⚠️ CUDA支持: ❌ 未启用")
        print(f"💻 当前模式: CPU推理")
        print(f"💡 建议: 安装CUDA版本JAX")
    
    return has_working_gpu, gpu_count

def main():
    """主函数"""
    has_gpu, gpu_count = smart_cuda_detection()
    
    print(f"\n" + "=" * 50)
    print("🎯 使用建议:")
    
    if has_gpu:
        print("✅ 您的环境配置完美！")
        print("🚀 可以开始运行GPU加速的GPT推理了:")
        print("   python run_benchmark.py --max-samples 3 --show-gpu-info")
        print("   python main.py --max-samples 5")
        
        if gpu_count >= 4:
            print("⚡ 多GPU性能模式:")
            print("   python run_benchmark.py --max-samples 10")
    else:
        print("💡 建议安装CUDA版本以获得更好性能:")
        print("   pip uninstall jax jaxlib -y")
        print("   pip install jax[cuda11_pip]==0.6.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

if __name__ == "__main__":
    main()
