# 结果目录

此目录用于存储 GPT-1.5B JAX 推理项目的基准测试结果。

## 结果文件类型

### JSON 格式结果

- `benchmark_results_YYYYMMDD_HHMMSS.json` - 详细的基准测试结果
- `gpu_metrics_YYYYMMDD_HHMMSS.json` - GPU 使用情况和性能指标

### 可视化结果

- `performance_chart_YYYYMMDD_HHMMSS.png` - 性能图表
- `gpu_utilization_YYYYMMDD_HHMMSS.png` - GPU 利用率图表

### 日志文件

- `benchmark_YYYYMMDD_HHMMSS.log` - 详细的测试日志

## 结果内容说明

### benchmark_results.json 结构

```json
{
  "timestamp": "2024-XX-XX XX:XX:XX",
  "system_info": {
    "gpu_count": 4,
    "gpu_models": ["RTX 3080", "RTX 3080", "RTX 3080", "RTX 3080"],
    "jax_version": "0.6.1",
    "cuda_version": "11.8"
  },
  "model_config": {
    "n_layer": 48,
    "n_head": 25,
    "n_embd": 1600,
    "vocab_size": 50257
  },
  "benchmark_results": {
    "config_0": {
      "avg_throughput_tokens_per_sec": 1234.5,
      "avg_latency_ms": 123.4,
      "memory_usage_gb": 12.3,
      "samples_processed": 10
    }
  }
}
```

### GPU 指标文件结构

```json
{
  "timestamp": "2024-XX-XX XX:XX:XX",
  "gpu_metrics": {
    "gpu_0": {
      "utilization_percent": 95.2,
      "memory_used_gb": 10.1,
      "memory_total_gb": 12.0,
      "temperature_c": 72
    }
  }
}
```

## 查看结果

### 命令行查看

```bash
# 查看最新结果
ls -la results/ | tail -10

# 查看JSON结果
cat results/benchmark_results_latest.json | jq '.'

# 使用Makefile生成报告
make report
```

### 程序化访问

```python
import json
from pathlib import Path

# 读取最新结果
results_dir = Path('results')
latest_result = sorted(results_dir.glob('benchmark_results_*.json'))[-1]

with open(latest_result, 'r') as f:
    data = json.load(f)
    print(f"平均吞吐量: {data['benchmark_results']['config_0']['avg_throughput_tokens_per_sec']} tokens/sec")
```

## 结果分析

运行基准测试后，您可以：

1. **性能对比**：比较不同配置的推理性能
2. **资源使用**：分析 GPU 内存和计算资源利用率
3. **优化建议**：根据结果调整模型配置和系统参数
4. **趋势分析**：跟踪不同时间的性能变化

## 注意事项

- 结果文件会随时间累积，定期清理旧文件
- 大文件可能会影响 Git 仓库大小，建议添加到.gitignore
- 重要结果建议备份到云存储或其他位置
