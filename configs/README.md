# 配置文件目录

# 存放项目的各种配置文件

## 文件说明

- `default_config.py` - 默认配置文件，包含模型、基准测试和系统配置
- 可以添加其他自定义配置文件，如：
  - `production_config.py` - 生产环境配置
  - `debug_config.py` - 调试配置
  - `custom_model_config.py` - 自定义模型配置

## 使用方法

在代码中导入配置：

```python
from configs.default_config import DEFAULT_CONFIG, QUICK_TEST_CONFIG
```
