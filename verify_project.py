#!/usr/bin/env python3
"""
项目完整性验证脚本
检查所有模块是否可以正常导入和基本功能是否正常
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_environment():
    """检查Python环境"""
    print("🔍 检查Python环境...")
    print(f"Python版本: {sys.version}")
    print(f"项目路径: {project_root}")
    
    # 检查必要的包
    required_packages = ['jax', 'flax', 'numpy', 'json', 'pathlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少必要包: {', '.join(missing_packages)}")
        print("请运行: make install 或 pip install -r requirements.txt")
        return False
    
    return True


def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/data', 
        'src/inference',
        'src/utils',
        'datasets',
        'configs',
        'results',
        'tests',
        'scripts'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'environment.yml',
        'README.md',
        'Makefile',
        'setup_environment.sh',
        'src/models/gpt_model.py',
        'src/data/dataset_loader.py',
        'src/inference/benchmark.py',
        'src/utils/gpu_utils.py',
        'src/utils/results.py',
        'tests/test_all.py',
        'configs/default_config.py'
    ]
    
    missing_items = []
    
    # 检查目录
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ 目录 {dir_path}")
        else:
            print(f"❌ 目录 {dir_path}")
            missing_items.append(dir_path)
    
    # 检查文件
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ 文件 {file_path}")
        else:
            print(f"❌ 文件 {file_path}")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n⚠️ 缺少项目文件: {len(missing_items)} 个")
        return False
    
    return True


def check_imports():
    """检查模块导入"""
    print("\n📦 检查模块导入...")
    
    test_imports = [
        ('src.models.gpt_model', 'GPTConfig'),
        ('src.data.dataset_loader', 'DatasetLoader'),
        ('src.inference.benchmark', 'InferenceBenchmark'),
        ('src.utils.gpu_utils', 'setup_jax_environment'),
        ('src.utils.results', 'ResultManager'),
        ('configs.default_config', 'DEFAULT_CONFIG')
    ]
    
    import_errors = []
    
    for module, item in test_imports:
        try:
            exec(f"from {module} import {item}")
            print(f"✅ {module}.{item}")
        except ImportError as e:
            print(f"❌ {module}.{item} - {e}")
            import_errors.append(f"{module}.{item}")
        except Exception as e:
            print(f"⚠️ {module}.{item} - {e}")
    
    if import_errors:
        print(f"\n⚠️ 导入错误: {len(import_errors)} 个")
        return False
    
    return True


def check_datasets():
    """检查数据集文件"""
    print("\n📊 检查数据集文件...")
    
    datasets_dir = project_root / 'datasets'
    if not datasets_dir.exists():
        print("❌ datasets目录不存在")
        return False
    
    config_files = list(datasets_dir.glob('benchmark_dataset_config_*.jsonl'))
    print(f"找到 {len(config_files)} 个配置文件")
    
    if len(config_files) == 0:
        print("⚠️ 没有找到数据集配置文件")
        return False
    
    # 检查文件是否为空
    empty_files = []
    for config_file in config_files:
        if config_file.stat().st_size == 0:
            empty_files.append(config_file.name)
        else:
            print(f"✅ {config_file.name}")
    
    if empty_files:
        print(f"⚠️ 空文件: {', '.join(empty_files)}")
    
    return len(config_files) > len(empty_files)


def check_jax_installation():
    """检查JAX安装和GPU支持"""
    print("\n🚀 检查JAX安装...")
    
    try:
        import jax
        import jax.numpy as jnp
        print(f"✅ JAX版本: {jax.__version__}")
        
        # 检查设备
        devices = jax.devices()
        print(f"✅ 可用设备数量: {len(devices)}")
        
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        cpu_devices = [d for d in devices if d.device_kind == 'cpu']
        
        print(f"GPU设备: {len(gpu_devices)}")
        print(f"CPU设备: {len(cpu_devices)}")
        
        if gpu_devices:
            print("✅ GPU支持已启用")
            for i, device in enumerate(gpu_devices):
                print(f"  GPU {i}: {device}")
        else:
            print("⚠️ 未检测到GPU设备")
        
        # 简单计算测试
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print(f"✅ JAX计算测试通过: {y}")
        
        return True
        
    except ImportError as e:
        print(f"❌ JAX导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ JAX测试失败: {e}")
        return False


def main():
    """主验证流程"""
    print("🔍 GPT-1.5B JAX推理项目完整性验证")
    print("=" * 50)
    
    checks = [
        ("环境检查", check_environment),
        ("项目结构", check_project_structure), 
        ("模块导入", check_imports),
        ("数据集文件", check_datasets),
        ("JAX安装", check_jax_installation)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 20)
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 验证结果总结:")
    
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项通过")
    
    if passed == len(results):
        print("🎉 所有检查通过！项目已准备就绪。")
        print("\n下一步:")
        print("1. 运行测试: make test")
        print("2. 运行基准测试: make benchmark")
        print("3. 上传到GitHub: make upload-github")
        return True
    else:
        print("⚠️ 部分检查失败，请修复后重新验证。")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
