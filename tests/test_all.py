#!/usr/bin/env python3
"""
GPT-1.5B JAX推理项目测试套件
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestEnvironment(unittest.TestCase):
    """测试环境设置"""
    
    def test_jax_installation(self):
        """测试JAX是否正确安装"""
        self.assertTrue(JAX_AVAILABLE, "JAX未安装或导入失败")
        if JAX_AVAILABLE:
            print(f"JAX版本: {jax.__version__}")
    
    def test_gpu_availability(self):
        """测试GPU是否可用"""
        if JAX_AVAILABLE:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.device_kind == 'gpu']
            print(f"可用GPU数量: {len(gpu_devices)}")
            # 不强制要求GPU，因为可能在CPU环境测试
    
    def test_project_structure(self):
        """测试项目结构"""
        required_dirs = ['src', 'datasets', 'configs', 'results', 'tests']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(dir_path.exists(), f"目录 {dir_name} 不存在")
        
        required_files = ['main.py', 'requirements.txt', 'README.md']
        for file_name in required_files:
            file_path = project_root / file_name
            self.assertTrue(file_path.exists(), f"文件 {file_name} 不存在")


class TestDatasetLoader(unittest.TestCase):
    """测试数据集加载器"""
    
    def setUp(self):
        """设置测试环境"""
        if JAX_AVAILABLE:
            from src.data.dataset_loader import DatasetLoader, SimpleTokenizer
            self.dataset_loader = DatasetLoader(str(project_root / 'datasets'))
            self.tokenizer = SimpleTokenizer()
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX不可用")
    def test_dataset_loading(self):
        """测试数据集加载"""
        valid_datasets = self.dataset_loader.get_valid_datasets()
        self.assertGreater(len(valid_datasets), 0, "没有找到有效的数据集")
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX不可用")
    def test_tokenizer(self):
        """测试分词器"""
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text, max_length=20)
        decoded = self.tokenizer.decode(tokens)
        
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 20)  # 应该填充到指定长度
        self.assertIsInstance(decoded, str)


class TestGPTModel(unittest.TestCase):
    """测试GPT模型"""
    
    def setUp(self):
        """设置测试环境"""
        if JAX_AVAILABLE:
            from src.models.gpt_model import GPTConfig, GPT, create_device_mesh
            self.config = GPTConfig(
                vocab_size=1000,  # 减小词汇表大小以加快测试
                n_positions=128,  # 减小序列长度
                n_embd=64,       # 减小嵌入维度
                n_layer=2,       # 减少层数
                n_head=2         # 减少注意力头数
            )
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX不可用")
    def test_model_creation(self):
        """测试模型创建"""
        from src.models.gpt_model import GPT
        model = GPT(self.config)
        self.assertIsNotNone(model)
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX不可用")
    def test_device_mesh_creation(self):
        """测试设备网格创建"""
        from src.models.gpt_model import create_device_mesh
        devices = jax.devices()
        if len(devices) > 0:
            device_count = min(len(devices), 2)  # 最多使用2个设备进行测试
            mesh = create_device_mesh(device_count)
            self.assertIsNotNone(mesh)


class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX不可用")
    def test_gpu_utils(self):
        """测试GPU工具"""
        try:
            from src.utils.gpu_utils import get_gpu_info, monitor_gpu_usage
            gpu_info = get_gpu_info()
            self.assertIsInstance(gpu_info, (dict, type(None)))
        except ImportError:
            # GPUtil可能未安装，这是可选的
            pass
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX不可用")
    def test_results_utils(self):
        """测试结果处理工具"""
        from src.utils.results import ResultManager
        result_manager = ResultManager("test_results")
        self.assertIsNotNone(result_manager)


def run_tests():
    """运行所有测试"""
    print("🧪 开始运行GPT-1.5B JAX推理项目测试套件")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestEnvironment,
        TestDatasetLoader,
        TestGPTModel,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ 所有测试通过！")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
        return False
    
    return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
