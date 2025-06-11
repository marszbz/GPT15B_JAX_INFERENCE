#!/usr/bin/env python3
"""
JAX版本检测脚本 - 准确检测CPU/GPU版本
"""

import jax
import jax.numpy as jnp

def check_jax_version():
    """检查JAX版本和设备信息"""
    print(f"JAX版本: {jax.__version__}")
    
    # 获取所有设备
    devices = jax.devices()
    print(f"总设备数: {len(devices)}")
    print("设备列表:")
    for i, device in enumerate(devices):
        print(f"  设备 {i}: {device}")
    
    # 正确的GPU/CPU设备检测
    gpu_devices = [d for d in devices if d.device_kind.lower() == 'gpu' or 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]
    cpu_devices = [d for d in devices if d.device_kind.lower() == 'cpu']
    
    print(f"\nGPU设备数: {len(gpu_devices)}")
    print(f"CPU设备数: {len(cpu_devices)}")
    
    # 详细设备信息
    if gpu_devices:
        print("\nGPU设备详情:")
        for i, device in enumerate(gpu_devices):
            print(f"  GPU {i}: {device}")
            print(f"    设备类型: {device.device_kind}")
            print(f"    平台: {device.platform}")
    
    if cpu_devices:
        print("\nCPU设备详情:")
        for i, device in enumerate(cpu_devices):
            print(f"  CPU {i}: {device}")
            print(f"    设备类型: {device.device_kind}")
            print(f"    平台: {device.platform}")
    
    # 判断JAX版本类型
    if len(gpu_devices) > 0:
        version_type = "GPU版本 (CUDA支持)"
        print(f"\n✅ JAX安装类型: {version_type}")
        
        # 测试GPU计算
        try:
            test_array = jnp.array([1.0, 2.0, 3.0, 4.0])
            with jax.default_device(gpu_devices[0]):
                result = jnp.sum(test_array)
            print(f"🧪 GPU计算测试: {result}")
            print("✅ GPU计算正常工作")
        except Exception as e:
            print(f"⚠️ GPU计算测试失败: {e}")
    else:
        version_type = "CPU版本"
        print(f"\n⚠️ JAX安装类型: {version_type}")
        print("💡 建议安装GPU版本以获得更好性能")
    
    # 检查JAXlib版本
    try:
        import jaxlib
        print(f"\nJAXlib版本: {jaxlib.__version__}")
        
        # 检查是否包含CUDA
        if 'cuda' in jaxlib.__version__.lower():
            print("✅ JAXlib包含CUDA支持")
        else:
            print("❌ JAXlib不包含CUDA支持")
    except ImportError:
        print("\n❌ JAXlib未安装")
    
    return len(gpu_devices) > 0, version_type

def main():
    print("🔍 JAX版本和设备检测")
    print("=" * 50)
    
    has_gpu, version_type = check_jax_version()
    
    print("\n" + "=" * 50)
    print("📋 检测结果:")
    
    if has_gpu:
        print("🎉 检测到GPU设备，JAX GPU版本工作正常！")
        print("🚀 可以使用GPU加速进行模型推理。")
    else:
        print("⚠️ 未检测到GPU设备，当前为CPU版本。")
        print("\n💡 如需GPU加速，请安装CUDA版本:")
        print("   pip uninstall jax jaxlib -y")
        print("   pip install jax[cuda11_pip]==0.6.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

if __name__ == "__main__":
    main()
