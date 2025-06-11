"""
数据集加载器 - 处理您的JSONL格式数据集
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class SimpleTokenizer:
    """简化的文本分词器"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """编码文本为token序列"""
        # 简化实现：基于字符的编码
        tokens = []
        for char in text.lower():
            token_id = min(ord(char), self.vocab_size - 1)
            tokens.append(token_id)
        
        # 处理长度限制
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                # 填充到指定长度
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """解码token序列为文本"""
        chars = []
        for token_id in tokens:
            if token_id != self.pad_token_id and token_id > 0:
                chars.append(chr(min(token_id, 127)))  # 限制在ASCII范围内
        return ''.join(chars)


class DatasetLoader:
    """数据集加载器 - 处理JSONL格式数据"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """加载所有数据集配置"""
        print("📁 加载数据集...")
        
        if not self.dataset_dir.exists():
            print(f"❌ 数据集目录不存在: {self.dataset_dir}")
            return
        
        # 查找所有配置文件
        config_files = list(self.dataset_dir.glob("benchmark_dataset_config_*.jsonl"))
        print(f"🔍 找到 {len(config_files)} 个配置文件")
        
        for config_file in config_files:
            config_id = config_file.stem.split('_')[-1]
            samples = []
            
            # 检查文件是否为空
            if config_file.stat().st_size == 0:
                print(f"⚠️ 配置 {config_id} 文件为空，跳过")
                continue
            
            # 读取JSONL文件
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                samples.append(sample)
                            except json.JSONDecodeError as e:
                                print(f"⚠️ 配置 {config_id} 第 {line_num} 行JSON解析错误: {e}")
                                continue
                
                if samples:
                    self.datasets[config_id] = samples
                    print(f"📊 配置 {config_id}: {len(samples)} 个样本")
                    
                    # 显示示例
                    if samples:
                        sample = samples[0]
                        print(f"   示例: prompt_length={sample.get('prompt_length', 'N/A')}, "
                              f"generation_length={sample.get('generation_length', 'N/A')}")
                else:
                    print(f"⚠️ 配置 {config_id} 没有有效样本")
                    
            except Exception as e:
                print(f"❌ 加载配置 {config_id} 失败: {e}")
    
    def get_valid_datasets(self) -> Dict[str, List[Dict]]:
        """获取所有有效数据集"""
        return {k: v for k, v in self.datasets.items() if v}
    
    def get_dataset_stats(self) -> Dict[str, Dict]:
        """获取数据集统计信息"""
        stats = {}
        for config_id, samples in self.datasets.items():
            if not samples:
                continue
                
            prompt_lengths = [s.get('prompt_length', 0) for s in samples]
            gen_lengths = [s.get('generation_length', 0) for s in samples]
            
            stats[config_id] = {
                'sample_count': len(samples),
                'avg_prompt_length': np.mean(prompt_lengths) if prompt_lengths else 0,
                'avg_generation_length': np.mean(gen_lengths) if gen_lengths else 0,
                'prompt_length_range': (min(prompt_lengths), max(prompt_lengths)) if prompt_lengths else (0, 0),
                'generation_length_range': (min(gen_lengths), max(gen_lengths)) if gen_lengths else (0, 0),
                'source_types': list(set(s.get('source_type', 'unknown') for s in samples))
            }
        
        return stats
    
    def get_dataset_by_config(self, config_id: str) -> List[Dict]:
        """获取指定配置的数据集"""
        return self.datasets.get(config_id, [])
    
    def print_dataset_summary(self):
        """打印数据集摘要"""
        stats = self.get_dataset_stats()
        print("\n📊 数据集摘要:")
        print("-" * 50)
        
        for config_id, stat in stats.items():
            print(f"配置 {config_id}:")
            print(f"  样本数: {stat['sample_count']}")
            print(f"  平均prompt长度: {stat['avg_prompt_length']:.1f}")
            print(f"  平均生成长度: {stat['avg_generation_length']:.1f}")
            print(f"  Prompt长度范围: {stat['prompt_length_range']}")
            print(f"  生成长度范围: {stat['generation_length_range']}")
            print(f"  数据源类型: {', '.join(stat['source_types'])}")
            print()
