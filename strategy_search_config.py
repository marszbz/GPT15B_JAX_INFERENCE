# GPT-1.5B 策略搜索配置文件
# 定义搜索空间、约束条件和优化目标

# 搜索空间配置
SEARCH_SPACE = {
    # Mesh拓扑配置
    "mesh_topologies": {
        "1d": {
            "shapes": [(1,), (2,), (4,), (8,)],
            "axis_names": [("data",), ("data",), ("data",), ("data",)]
        },
        "2d": {
            "shapes": [(1,2), (2,1), (2,2), (1,4), (4,1), (2,4), (4,2)],
            "axis_names": [("data","model")] * 7
        },
        "3d": {
            "shapes": [(2,2,2), (1,2,4), (2,1,4), (1,4,2)],
            "axis_names": [("pipeline","data","model")] * 4
        }
    },
    
    # 分片策略模板
    "sharding_templates": {
        "data_parallel": {
            "embedding": "PartitionSpec('data', None)",
            "attention_qkv": "PartitionSpec('data', None, None)",
            "attention_out": "PartitionSpec('data', None, None)",
            "mlp_up": "PartitionSpec('data', None, None)",
            "mlp_down": "PartitionSpec('data', None, None)",
            "lm_head": "PartitionSpec('data', None)",
            "input": "PartitionSpec('data', None)"
        },
        "model_parallel": {
            "embedding": "PartitionSpec(None, 'model')",
            "attention_qkv": "PartitionSpec(None, None, 'model')",
            "attention_out": "PartitionSpec(None, 'model', None)",
            "mlp_up": "PartitionSpec(None, None, 'model')",
            "mlp_down": "PartitionSpec(None, 'model', None)",
            "lm_head": "PartitionSpec(None, 'model')",
            "input": "PartitionSpec(None, None)"        },
        "hybrid": {
            "embedding": "PartitionSpec(None, 'model')",
            "attention_qkv": "PartitionSpec('data', None, 'model')",
            "attention_out": "PartitionSpec('data', 'model', None)",
            "mlp_up": "PartitionSpec('data', None, 'model')",
            "mlp_down": "PartitionSpec('data', 'model', None)",
            "lm_head": "PartitionSpec(None, 'model')",
            "input": "PartitionSpec('data', None)"
        },
        "pipeline_parallel": {
            "embedding": "PartitionSpec('pipeline', None, None)",
            "attention_qkv": "PartitionSpec('pipeline', 'data', 'model')",
            "attention_out": "PartitionSpec('pipeline', 'data', 'model')",
            "mlp_up": "PartitionSpec('pipeline', 'data', 'model')",
            "mlp_down": "PartitionSpec('pipeline', 'data', 'model')",
            "lm_head": "PartitionSpec('pipeline', None, None)",
            "input": "PartitionSpec(None, 'data', None)"
        },
        # RTX 3080优化策略 - 针对内存限制的特殊优化
        "rtx3080_optimized": {
            "embedding": "PartitionSpec(None, 'model')",  # 词汇表分片减少内存
            "attention_qkv": "PartitionSpec(None, None, 'model')",  # 头维度分片
            "attention_out": "PartitionSpec(None, 'model', None)",  # 输出分片
            "mlp_up": "PartitionSpec(None, None, 'model')",  # MLP强制分片
            "mlp_down": "PartitionSpec(None, 'model', None)",
            "lm_head": "PartitionSpec(None, 'model')",  # 输出头分片
            "input": "PartitionSpec('data', None)",  # 小批次数据并行
            "description": "专为RTX 3080内存限制优化的分片策略"
        },
        # 内存节约型策略
        "memory_efficient": {
            "embedding": "PartitionSpec(None, 'model')",
            "attention_qkv": "PartitionSpec(None, None, 'model')",
            "attention_out": "PartitionSpec(None, 'model', None)",
            "mlp_up": "PartitionSpec(None, None, 'model')",
            "mlp_down": "PartitionSpec(None, 'model', None)",
            "lm_head": "PartitionSpec(None, 'model')",
            "input": "PartitionSpec(None, None)",
            "description": "最大化内存节约的分片策略"
        }
    }
}

# 搜索约束
CONSTRAINTS = {
    # 内存约束 (GB per GPU) - 会根据硬件自动调整
    "max_memory_per_gpu": 20.0,  # 默认值，实际使用时会根据硬件配置调整
    
    # 针对不同GPU的特定约束
    "gpu_specific_constraints": {
        "rtx3080_quad": {
            "max_memory_per_gpu": 8.0,  # RTX 3080留出2GB给系统
            "max_batch_size": 32,       # 受内存限制
            "prefer_model_parallel": True,  # 优先模型并行
            "max_sequence_length": 1024     # 限制序列长度
        },
        "rtx3080_12gb_quad": {
            "max_memory_per_gpu": 10.0,
            "max_batch_size": 48,
            "prefer_model_parallel": True,
            "max_sequence_length": 1536
        },
        "rtx3090_quad": {
            "max_memory_per_gpu": 20.0,
            "max_batch_size": 64,
            "prefer_hybrid_parallel": True,
            "max_sequence_length": 2048
        },
        "rtx4090_quad": {
            "max_memory_per_gpu": 20.0,
            "max_batch_size": 80,
            "prefer_hybrid_parallel": True,
            "max_sequence_length": 2048
        },
        "a100_quad": {
            "max_memory_per_gpu": 35.0,
            "max_batch_size": 128,
            "prefer_data_parallel": True,
            "max_sequence_length": 4096
        }
    },
    
    # 通信约束
    "max_communication_ratio": 0.3,  # 通信时间不超过总时间的30%
    
    # 延迟约束
    "max_latency_ms": 500.0,
    
    # 最小批次大小
    "min_batch_size": 1,
    "max_batch_size": 64,  # 默认值，会被GPU特定约束覆盖
    
    # 设备利用率约束
    "min_device_utilization": 0.7,
    
    # 内存安全边距
    "memory_safety_margin": 0.1  # 保留10%内存作为安全边距
}

# 优化目标权重
OPTIMIZATION_OBJECTIVES = {
    "throughput": 0.4,        # 吞吐量权重
    "latency": 0.3,           # 延迟权重
    "memory_efficiency": 0.2,  # 内存效率权重
    "scalability": 0.1        # 可扩展性权重
}

# 搜索算法配置
SEARCH_ALGORITHMS = {
    "exhaustive": {
        "enabled": True,
        "max_strategies": 50,  # 限制策略数量避免过长时间
        "early_stopping": True,
        "patience": 10  # 连续10个策略无改进则停止
    },
    
    "genetic": {
        "enabled": True,
        "population_size": 20,
        "generations": 10,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "elite_ratio": 0.2
    },
    
    "bayesian": {
        "enabled": False,  # 需要额外依赖
        "n_initial": 10,
        "n_iterations": 20,
        "acquisition": "ei"  # expected improvement
    },
    
    "random": {
        "enabled": True,
        "n_samples": 30,
        "seed": 42
    }
}

# 基准测试配置
BENCHMARK_CONFIG = {
    "warmup_iterations": 3,
    "measurement_iterations": 5,
    "test_batch_sizes": [1, 4, 8, 16, 32],
    "test_sequence_lengths": [128, 512, 1024],
    "precision": "float32",  # or "float16", "bfloat16"
    
    # 性能测试参数
    "measure_memory": True,
    "measure_flops": True,
    "measure_communication": True,
    "profile_kernels": False  # 详细kernel分析（较慢）
}

# 模型配置变体
MODEL_VARIANTS = {
    "gpt15b_full": {
        "vocab_size": 50257,
        "n_positions": 2048,
        "n_embd": 1600,
        "n_layer": 48,
        "n_head": 25
    },
    "gpt15b_small": {  # 用于快速测试
        "vocab_size": 50257,
        "n_positions": 1024,
        "n_embd": 1600,
        "n_layer": 12,
        "n_head": 25
    },
    "gpt15b_micro": {  # 用于验证
        "vocab_size": 50257,
        "n_positions": 512,
        "n_embd": 768,
        "n_layer": 4,
        "n_head": 12
    }
}

# 硬件配置
HARDWARE_CONFIGS = {
    "rtx3080_quad": {
        "device_count": 4,
        "memory_per_device_gb": 10,  # RTX 3080 10GB版本
        "compute_capability": "8.6",
        "bandwidth_gbps": 760,
        "nvlink": False,
        "tensor_cores": True,
        "max_threads_per_block": 1024,
        "multiprocessor_count": 68
    },
    "rtx3080_12gb_quad": {
        "device_count": 4,
        "memory_per_device_gb": 12,  # RTX 3080 12GB版本
        "compute_capability": "8.6",
        "bandwidth_gbps": 912,
        "nvlink": False,
        "tensor_cores": True,
        "max_threads_per_block": 1024,
        "multiprocessor_count": 70
    },
    "rtx3090_quad": {
        "device_count": 4,
        "memory_per_device_gb": 24,
        "compute_capability": "8.6",
        "bandwidth_gbps": 936,
        "nvlink": False,
        "tensor_cores": True,
        "max_threads_per_block": 1024,
        "multiprocessor_count": 82
    },
    "rtx4090_quad": {
        "device_count": 4,
        "memory_per_device_gb": 24,
        "compute_capability": "8.9",
        "bandwidth_gbps": 1008,
        "nvlink": False,
        "tensor_cores": True,
        "max_threads_per_block": 1024,
        "multiprocessor_count": 128
    },
    "a100_quad": {
        "device_count": 4,
        "memory_per_device_gb": 40,
        "compute_capability": "8.0",
        "bandwidth_gbps": 1555,
        "nvlink": True,
        "tensor_cores": True,
        "max_threads_per_block": 1024,
        "multiprocessor_count": 108
    }
}

# 高级搜索选项
ADVANCED_OPTIONS = {
    # 自适应搜索
    "adaptive_search": {
        "enabled": True,
        "adjust_based_on_hardware": True,
        "dynamic_pruning": True,
        "learning_rate": 0.1
    },
    
    # 多目标优化
    "multi_objective": {
        "enabled": True,
        "pareto_frontier": True,
        "scalarization_method": "weighted_sum"  # or "tchebycheff"
    },
    
    # 分布式搜索
    "distributed_search": {
        "enabled": False,
        "num_workers": 1,
        "strategy_per_worker": 10
    },
    
    # 缓存和重用
    "caching": {
        "enabled": True,
        "cache_compiled_functions": True,
        "cache_performance_results": True,
        "cache_directory": "./strategy_cache"
    }
}

# 输出和报告配置
OUTPUT_CONFIG = {
    "save_all_results": True,
    "save_intermediate": True,
    "create_visualization": True,
    "export_best_strategy": True,
    "generate_code_template": True,
    
    "output_formats": ["json", "csv", "txt"],
    "include_metadata": True,
    "compress_results": False
}
