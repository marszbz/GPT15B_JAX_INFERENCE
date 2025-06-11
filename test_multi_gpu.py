#!/usr/bin/env python3
"""
快速多GPU测试脚本
直接运行多GPU分析，无需Jupyter笔记本
"""

import os
import sys
import time
import json
from pathlib import Path

# 设置JAX环境（必须在导入JAX之前）
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
# 清理可能存在的XLA_FLAGS
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

print("🔧 设置JAX环境变量...")
print(f"   XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
print(f"   XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")

try:
    # 导入JAX相关包
    import jax
    import jax.numpy as jnp
    from jax import random, devices
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils
    import numpy as np
    
    print(f"✅ JAX {jax.__version__} 导入成功")
    
    # 基本GPU检测
    print(f"\n🖥️ GPU环境检测:")
    print(f"   JAX版本: {jax.__version__}")
    print(f"   可用设备数量: {len(jax.devices())}")
    
    for i, device in enumerate(jax.devices()):
        print(f"   设备 {i}: {device}")
        print(f"     平台: {device.platform}")
        print(f"     设备类型: {device.device_kind}")
    
    # 检查CUDA支持
    if jax.devices()[0].platform == 'gpu':
        print("   ✅ 检测到CUDA支持")
    else:
        print("   ⚠️ 未检测到CUDA支持")
    
    # 尝试创建mesh
    print(f"\n🔗 尝试创建设备mesh:")
    devices_list = jax.devices()
    
    if len(devices_list) == 1:
        print("   ⚠️ 只有1个设备，无法进行并行化")
        mesh = None
    else:
        try:
            if len(devices_list) == 4:
                # 4个GPU: 尝试2x2 mesh
                mesh_shape = (2, 2)
                mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
                mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
                print(f"   ✅ 成功创建2x2 mesh，轴名称: ('data', 'model')")
            elif len(devices_list) == 2:
                mesh_shape = (2, 1)
                mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
                mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
                print(f"   ✅ 成功创建2x1 mesh，轴名称: ('data', 'model')")
            else:
                # 1D mesh
                mesh_devices = np.array(devices_list).reshape(-1, 1)
                mesh = Mesh(mesh_devices, axis_names=('data',))
                print(f"   ✅ 成功创建1D mesh，设备数: {len(devices_list)}")
            
            print(f"   Mesh形状: {mesh.shape}")
            print(f"   Mesh轴名称: {mesh.axis_names}")
            
        except Exception as e:
            print(f"   ❌ Mesh创建失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            mesh = None
    
    # 简单的张量操作测试
    print(f"\n🧪 基本张量操作测试:")
    try:
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (4, 128, 768))  # 模拟GPT输入
        print(f"   测试张量形状: {x.shape}")
        
        # 简单的矩阵乘法
        start_time = time.time()
        y = jnp.matmul(x, x.transpose(0, 2, 1))
        jax.block_until_ready(y)
        compute_time = time.time() - start_time
        
        print(f"   矩阵乘法结果形状: {y.shape}")
        print(f"   计算时间: {compute_time*1000:.2f}ms")
        print("   ✅ 基本张量操作正常")
        
    except Exception as e:
        print(f"   ❌ 张量操作测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 内存使用情况检查
    print(f"\n💾 内存使用分析:")
    
    # 估算参数内存（1.5B参数模型）
    param_count_1_5b = 1500000000  # 1.5B参数
    param_memory_fp32 = param_count_1_5b * 4 / (1024**3)  # GB
    param_memory_fp16 = param_memory_fp32 / 2
    
    print(f"   1.5B参数模型内存需求:")
    print(f"     FP32: {param_memory_fp32:.2f} GB")
    print(f"     FP16: {param_memory_fp16:.2f} GB")
    
    # RTX 3090内存分析
    rtx_3090_memory = 24  # GB
    num_gpus = len(jax.devices())
    
    print(f"   GPU内存分析:")
    print(f"     单个RTX 3090内存: {rtx_3090_memory} GB")
    print(f"     总GPU内存: {rtx_3090_memory * num_gpus} GB")
    print(f"     内存利用率(FP32): {(param_memory_fp32/rtx_3090_memory)*100:.1f}%")
    print(f"     内存利用率(FP16): {(param_memory_fp16/rtx_3090_memory)*100:.1f}%")
    
    # 并行化建议
    print(f"\n💡 并行化建议:")
    if mesh is not None:
        if param_memory_fp32 < rtx_3090_memory:
            print("   ✅ 推荐使用数据并行")
            print("   - 每个GPU运行完整模型")
            print("   - 在batch维度上分片")
        else:
            print("   ✅ 推荐使用模型并行")
            print("   - 将模型参数分布到多个GPU")
            print("   - 在参数维度上分片")
    else:
        print("   ⚠️ 无法创建mesh，需要修复多GPU设置")
    
    # 生成诊断报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jax_version": jax.__version__,
        "gpu_count": len(jax.devices()),
        "devices": [str(d) for d in jax.devices()],
        "cuda_support": jax.devices()[0].platform == 'gpu',
        "mesh_created": mesh is not None,
        "mesh_shape": mesh.shape if mesh else None,
        "mesh_axes": mesh.axis_names if mesh else None,
        "memory_analysis": {
            "model_1_5b_fp32_gb": param_memory_fp32,
            "model_1_5b_fp16_gb": param_memory_fp16,
            "gpu_memory_gb": rtx_3090_memory,
            "total_gpu_memory_gb": rtx_3090_memory * num_gpus,
            "memory_utilization_fp32": (param_memory_fp32/rtx_3090_memory)*100,
            "memory_utilization_fp16": (param_memory_fp16/rtx_3090_memory)*100
        }
    }
    
    # 保存报告
    report_file = Path("multi_gpu_diagnosis.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 诊断报告已保存: {report_file}")
    
    # 总结
    print(f"\n🎯 诊断总结:")
    print(f"   GPU数量: {len(jax.devices())}")
    print(f"   CUDA支持: {'✅' if jax.devices()[0].platform == 'gpu' else '❌'}")
    print(f"   Mesh创建: {'✅' if mesh else '❌'}")
    
    if mesh:
        print(f"   🚀 多GPU并行化就绪!")
        print(f"   📋 下一步: 实现具体的分片策略")
    else:
        print(f"   ⚠️ 需要修复mesh创建问题")
        print(f"   📋 下一步: 调试mesh创建代码")

except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    print("💡 请确保已正确安装JAX with CUDA支持")
    print("   conda activate jax-env")
    print("   pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

except Exception as e:
    print(f"❌ 意外错误: {e}")
    import traceback
    traceback.print_exc()

print(f"\n✅ 多GPU诊断完成!")
