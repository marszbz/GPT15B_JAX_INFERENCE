#!/usr/bin/env python3
"""
安全的JAX环境设置脚本
避免XLA标志冲突
"""

import os
import sys

def setup_safe_jax_env():
    """设置安全的JAX环境变量"""
    print("🔧 配置JAX环境...")
    
    # 清理可能冲突的环境变量
    if 'XLA_FLAGS' in os.environ:
        del os.environ['XLA_FLAGS']
        print("✅ 清理了XLA_FLAGS")
    
    # 设置推荐的环境变量
    safe_env = {
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.8',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3',  # 使用所有4个GPU
    }
    
    for key, value in safe_env.items():
        os.environ[key] = value
        print(f"✅ 设置 {key}={value}")
    
    print("🎉 JAX环境配置完成！")

def test_jax_import():
    """测试JAX导入"""
    try:
        import jax
        import jax.numpy as jnp
        print(f"✅ JAX {jax.__version__} 导入成功")
        
        devices = jax.devices()
        print(f"✅ 检测到 {len(devices)} 个设备")
        
        # 简单计算测试
        x = jnp.array([1, 2, 3])
        result = jnp.sum(x)
        print(f"✅ 计算测试通过: {result}")
        
        return True
    except Exception as e:
        print(f"❌ JAX测试失败: {e}")
        return False

def main():
    print("🚀 安全JAX环境设置")
    print("=" * 40)
    
    setup_safe_jax_env()
    
    print("\n🧪 测试JAX...")
    if test_jax_import():
        print("\n✅ JAX环境配置成功！")
        print("🚀 现在可以运行:")
        print("   python run_benchmark.py --max-samples 3")
    else:
        print("\n❌ JAX环境配置失败")
        print("请检查CUDA和JAX安装")

if __name__ == "__main__":
    main()
