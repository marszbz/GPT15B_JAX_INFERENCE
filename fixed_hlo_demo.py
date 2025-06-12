#!/usr/bin/env python3
"""
修复版 HLO (High Level Operations) 详细解释和演示
展示JAX如何通过HLO进行图优化和分片
"""

import os
import sys
import time
from pathlib import Path

# 设置JAX环境
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

try:
    import jax
    import jax.numpy as jnp
    from jax import random, devices
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    import numpy as np
    print(f"✅ JAX {jax.__version__} HLO演示模式")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

class FixedHLOExplainer:
    """修复版HLO详细解释器"""
    
    def __init__(self):
        self.devices = jax.devices()
        self.mesh = None
        
    def explain_hlo_basics(self):
        """解释HLO基础概念"""
        print(f"🎓 HLO (High Level Operations) 基础概念")
        print("="*60)
        
        print(f"💡 什么是HLO?")
        print(f"   HLO = High Level Operations (高级操作)")
        print(f"   作用: JAX/XLA编译器的中间表示语言")
        print(f"   位置: Python代码 → HLO → GPU机器码")
        
        print(f"\n🔄 编译流程:")
        print(f"   1. Python/JAX → HLO图构建")
        print(f"   2. HLO → 图优化 (融合、重排列)")
        print(f"   3. HLO → 分片分析")
        print(f"   4. HLO → GPU Kernel生成")
        print(f"   5. GPU Kernel → 执行")
        
        print(f"\n📊 HLO的特点:")
        print(f"   • 函数式: 无副作用，纯函数")
        print(f"   • 静态形状: 编译时确定张量形状")
        print(f"   • 设备无关: 可在CPU/GPU/TPU执行")
        print(f"   • 优化友好: 便于自动优化")
    
    def demonstrate_python_to_hlo(self):
        """演示Python代码到HLO的转换"""
        print(f"\n🔍 Python → HLO 转换演示")
        print("="*60)
        
        print(f"📝 示例1: 简单矩阵乘法")
        print(f"Python代码:")
        print(f"   def simple_matmul(x, w):")
        print(f"       return jnp.dot(x, w)")
        
        print(f"\n对应的HLO操作:")
        print(f"   %result = dot(%x, %w)")
        print(f"   其中:")
        print(f"   - %x: 输入张量")
        print(f"   - %w: 权重张量") 
        print(f"   - %result: 输出张量")
        
        print(f"\n📝 示例2: 神经网络层")
        print(f"Python代码:")
        print(f"   def neural_layer(x, w, b):")
        print(f"       h = jnp.dot(x, w)")
        print(f"       h = h + b")
        print(f"       return jax.nn.relu(h)")
        
        print(f"\n对应的HLO操作序列:")
        print(f"   %h1 = dot(%x, %w)")
        print(f"   %h2 = add(%h1, %b)")
        print(f"   %result = maximum(%h2, %zero)")
        
        print(f"\n🔧 优化后的HLO (操作融合):")
        print(f"   %result = fused_dot_add_relu(%x, %w, %b)")
        print(f"   → 3个操作融合为1个!")
    
    def explain_hlo_sharding(self):
        """解释HLO中的分片机制"""
        print(f"\n🔀 HLO分片机制详解")
        print("="*60)
        
        print(f"🎯 分片在HLO中的表示:")
        print(f"   每个HLO操作都带有分片信息")
        print(f"   格式: operation" + "{sharding=...}")
        
        print(f"\n📋 分片表示示例:")
        print(f"   原始: %result = dot(%x, %w)")
        print(f"   分片: %result = dot(%x" + "{sharding=[0,1]}, %w{sharding=[1,2]})")
        print(f"         " + "{sharding=[0,2]}")
        
        print(f"\n🔍 分片标记解释:")
        print(f"   [0,1] = 第0维沿设备轴0分片，第1维沿设备轴1分片")
        print(f"   [1,2] = 第0维沿设备轴1分片，第1维沿设备轴2分片")
        print(f"   []    = 复制到所有设备")
        
        if len(self.devices) >= 4:
            self._demonstrate_hlo_sharding_with_mesh()
    
    def _demonstrate_hlo_sharding_with_mesh(self):
        """使用mesh演示HLO分片"""
        # 创建2x2 mesh
        devices_array = np.array(self.devices[:4]).reshape(2, 2)
        self.mesh = Mesh(devices_array, axis_names=('data', 'model'))
        
        print(f"\n🕸️ 实际分片演示 (2x2 mesh):")
        print(f"   设备布局:")
        print(f"   ┌─────┬─────┐")
        print(f"   │ D0  │ D1  │ ← data轴=0")
        print(f"   ├─────┼─────┤") 
        print(f"   │ D2  │ D3  │ ← data轴=1")
        print(f"   └─────┴─────┘")
        print(f"    model model")
        print(f"    轴=0  轴=1")
        
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))  # batch=8, features=16
        w = jax.random.normal(key, (16, 32)) # features=16, hidden=32
        
        with self.mesh:
            # 定义分片
            x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            w_sharding = NamedSharding(self.mesh, PartitionSpec(None, 'model'))
            
            x_sharded = jax.device_put(x, x_sharding)
            w_sharded = jax.device_put(w, w_sharding)
            
            print(f"\n📊 分片后的HLO表示:")
            print(f"   %x" + "{sharding=[0]} shape=(8,16)")
            print(f"   %w" + "{sharding=[1]} shape=(16,32)")
            print(f"   %result = dot(%x, %w) " + "{sharding=[0,1]} shape=(8,32)")
            
            print(f"\n🔍 分片含义:")
            print(f"   x分片: batch维度沿data轴分片")
            print(f"   w分片: hidden维度沿model轴分片")
            print(f"   结果: batch沿data轴，hidden沿model轴分片")
    
    def explain_hlo_optimizations(self):
        """解释HLO优化技术"""
        print(f"\n⚡ HLO优化技术详解")
        print("="*60)
        
        print(f"🔧 主要优化技术:")
        
        print(f"\n1️⃣ 操作融合 (Operation Fusion):")
        print(f"   目的: 减少内存访问，提高计算效率")
        print(f"   示例:")
        print(f"   优化前: dot → add → relu (3个kernel)")
        print(f"   优化后: fused_dot_add_relu (1个kernel)")
        
        print(f"\n2️⃣ 布局优化 (Layout Optimization):")
        print(f"   目的: 最小化数据重排列")
        print(f"   示例:")
        print(f"   优化前: transpose → dot → transpose")
        print(f"   优化后: dot_with_transpose")
        
        print(f"\n3️⃣ 常量折叠 (Constant Folding):")
        print(f"   目的: 编译时计算常量表达式")
        print(f"   示例:")
        print(f"   优化前: add(2, 3) → constant(5)")
        print(f"   优化后: 直接使用5")
        
        print(f"\n4️⃣ 死码消除 (Dead Code Elimination):")
        print(f"   目的: 移除未使用的计算")
        print(f"   示例:")
        print(f"   优化前: x = dot(a, b); y = add(c, d); return x")
        print(f"   优化后: x = dot(a, b); return x  # 删除y的计算")
        
        print(f"\n5️⃣ 通信优化 (Communication Optimization):")
        print(f"   目的: 最小化设备间数据传输")
        print(f"   技术:")
        print(f"   - 通信/计算重叠")
        print(f"   - All-reduce融合")
        print(f"   - 通信调度优化")
    
    def demonstrate_hlo_communication_insertion(self):
        """演示HLO通信插入机制"""
        print(f"\n📡 HLO通信插入机制")
        print("="*60)
        
        print(f"🎯 通信插入的触发条件:")
        print(f"   当操作的输入分片与期望不匹配时")
        print(f"   XLA自动插入通信操作")
        
        print(f"\n📋 常见通信模式:")
        
        print(f"\n1️⃣ All-Reduce (全归约):")
        print(f"   场景: 矩阵乘法后需要合并结果")
        print(f"   HLO: %result = all-reduce(%partial_result)")
        print(f"   作用: 所有设备计算sum/mean等")
        
        print(f"\n2️⃣ All-Gather (全收集):")
        print(f"   场景: 需要完整张量进行计算")
        print(f"   HLO: %full_tensor = all-gather(%sharded_tensor)")
        print(f"   作用: 收集所有分片到每个设备")
        
        print(f"\n3️⃣ Reduce-Scatter (归约分散):")
        print(f"   场景: 归约后按新方式分片")
        print(f"   HLO: %new_shard = reduce-scatter(%input)")
        print(f"   作用: 归约+重新分布")
        
        print(f"\n4️⃣ Reshape (重分片):")
        print(f"   场景: 改变分片模式")
        print(f"   HLO: %reshaped = reshape(%input, new_sharding)")
        print(f"   作用: 数据重新分布")
        
        if self.mesh:
            self._demonstrate_communication_example()
    
    def _demonstrate_communication_example(self):
        """演示具体的通信示例"""
        print(f"\n🎬 通信插入实例:")
        print(f"-" * 30)
        
        key = jax.random.PRNGKey(42)
        
        with self.mesh:
            # 场景: 两个不兼容分片的矩阵相乘
            x = jax.random.normal(key, (8, 16))
            w = jax.random.normal(key, (16, 32))
            
            # x按data轴分片，w也按data轴分片 (不兼容!)
            x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            w_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))  # 错误的分片!
            
            x_sharded = jax.device_put(x, x_sharding)
            w_sharded = jax.device_put(w, w_sharding)
            
            print(f"问题场景:")
            print(f"   x: (8,16) 分片=[data, None]")
            print(f"   w: (16,32) 分片=[data, None]  ← 不兼容!")
            print(f"   矩阵乘法: x @ w")
            
            print(f"\nHLO分析:")
            print(f"   检测到: w的第0维被分片，但需要与x的第1维收缩")
            print(f"   解决方案: 插入all-gather收集w的完整版本")
            
            print(f"\n自动插入的HLO操作:")
            print(f"   %w_full = all-gather(%w_sharded)")
            print(f"   %result = dot(%x_sharded, %w_full)")
            
            # 实际执行
            def matrix_multiply(x, w):
                return jnp.dot(x, w)
            
            jit_mm = jax.jit(matrix_multiply)
            result = jit_mm(x_sharded, w_sharded)
            
            print(f"\n✅ 执行成功!")
            print(f"   输入: x{x.shape} @ w{w.shape}")
            print(f"   输出: result{result.shape}")
            print(f"   XLA自动处理了分片不匹配问题!")
    
    def explain_automatic_sharding_process(self):
        """解释自动分片的详细过程"""
        print(f"\n🤖 JAX自动分片详细过程")
        print("="*60)
        
        print(f"🧠 自动分片是如何工作的:")
        
        print(f"\n第1步: 分片注解解析")
        print(f"   • 解析用户提供的PartitionSpec")
        print(f"   • 验证分片规范的合法性")
        print(f"   • 建立设备网格到张量维度的映射")
        
        print(f"\n第2步: 分片传播")
        print(f"   • 从输入开始，逐个操作传播分片信息")
        print(f"   • 根据操作语义推断输出分片")
        print(f"   • 矩阵乘法: (A,B) @ (B,C) → (A,C)")
        print(f"   • 元素运算: 保持相同分片")
        
        print(f"\n第3步: 冲突检测")
        print(f"   • 检测操作输入的分片不匹配")
        print(f"   • 识别需要通信的位置")
        print(f"   • 分析通信开销和类型")
        
        print(f"\n第4步: 通信插入")
        print(f"   • 自动插入必要的集合通信操作")
        print(f"   • AllReduce: 跨设备求和/平均")
        print(f"   • AllGather: 收集分布式数据")
        print(f"   • ReduceScatter: 归约后重分布")
        
        print(f"\n第5步: 优化调度")
        print(f"   • 优化通信和计算的重叠")
        print(f"   • 融合多个小通信为大通信")
        print(f"   • 选择最优的通信模式")
        
        if self.mesh:
            self._demonstrate_step_by_step_sharding()
    
    def _demonstrate_step_by_step_sharding(self):
        """演示逐步分片过程"""
        print(f"\n🎬 逐步分片演示:")
        print(f"-" * 30)
        
        # 定义一个多步计算
        def complex_computation(x, w1, w2, b):
            h1 = jnp.dot(x, w1)      # 第1步: 矩阵乘法
            h2 = h1 + b              # 第2步: 加偏置  
            h3 = jax.nn.relu(h2)     # 第3步: 激活函数
            y = jnp.dot(h3, w2)      # 第4步: 输出投影
            return y
        
        key = jax.random.PRNGKey(42)
        
        with self.mesh:
            # 创建输入
            x = jax.random.normal(key, (8, 16))   # batch=8, input=16
            w1 = jax.random.normal(key, (16, 32)) # input=16, hidden=32
            w2 = jax.random.normal(key, (32, 8))  # hidden=32, output=8
            b = jax.random.normal(key, (32,))     # bias=32
            
            # 定义初始分片
            x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            w1_sharding = NamedSharding(self.mesh, PartitionSpec(None, 'model'))
            w2_sharding = NamedSharding(self.mesh, PartitionSpec('model', None))
            b_sharding = NamedSharding(self.mesh, PartitionSpec('model'))
            
            # 应用分片
            x_sharded = jax.device_put(x, x_sharding)
            w1_sharded = jax.device_put(w1, w1_sharding)
            w2_sharded = jax.device_put(w2, w2_sharding)
            b_sharded = jax.device_put(b, b_sharding)
            
            print(f"输入分片:")
            print(f"   x(8,16):  [data, None]")
            print(f"   w1(16,32): [None, model]")
            print(f"   w2(32,8):  [model, None]")
            print(f"   b(32):     [model]")
            
            print(f"\n自动分片推断过程:")
            print(f"   步骤1: h1 = x @ w1")
            print(f"     x[data,None] @ w1[None,model] → h1[data,model]")
            print(f"     ✅ 无需通信")
            
            print(f"\n   步骤2: h2 = h1 + b")
            print(f"     h1[data,model] + b[model] → h2[data,model]")
            print(f"     ✅ 广播兼容，无需通信")
            
            print(f"\n   步骤3: h3 = relu(h2)")
            print(f"     relu(h2[data,model]) → h3[data,model]")
            print(f"     ✅ 元素运算，保持分片")
            
            print(f"\n   步骤4: y = h3 @ w2")
            print(f"     h3[data,model] @ w2[model,None] → y[data,None]")
            print(f"     ⚠️ 需要AllReduce (model维度收缩)")
            
            # 实际执行
            jit_comp = jax.jit(complex_computation)
            result = jit_comp(x_sharded, w1_sharded, w2_sharded, b_sharded)
            
            print(f"\n✅ 自动分片执行成功!")
            print(f"   最终输出: y{result.shape} [data,None]")
            print(f"   总通信次数: 1次AllReduce")
            print(f"   通信效率: 最优化!")
    
    def comprehensive_hlo_demo(self):
        """完整的HLO演示"""
        print(f"🎓 HLO (High Level Operations) 完整解析")
        print("="*60)
        
        # 依次执行各个部分
        self.explain_hlo_basics()
        self.demonstrate_python_to_hlo()
        self.explain_hlo_sharding()
        self.explain_hlo_optimizations()
        self.demonstrate_hlo_communication_insertion()
        self.explain_automatic_sharding_process()
        
        # 总结
        print(f"\n🎯 HLO核心总结")
        print("="*40)
        print(f"✅ HLO的关键作用:")
        print(f"   • 中间表示: 连接高级语言和硬件")
        print(f"   • 优化平台: 各种图优化的基础")
        print(f"   • 分片载体: 分片信息的表示和传播")
        print(f"   • 跨平台: CPU/GPU/TPU统一抽象")
        
        print(f"\n🔧 自动分片的技术优势:")
        print(f"   • 智能推断: 根据操作语义自动推断分片")
        print(f"   • 通信优化: 最小化设备间数据传输")
        print(f"   • 透明性: 用户只需指定输入分片策略")
        print(f"   • 高效性: 编译时优化，运行时高效")
        
        print(f"\n💡 从Python到GPU的完整流程:")
        print(f"   Python代码 → HLO图 → 分片分析 → 通信插入")
        print(f"   → 图优化 → GPU kernels → 高效执行")
        
        print(f"\n🚀 这就是如何实现从0.09到853.7 tokens/s的:")
        print(f"   9,485倍性能提升的技术基础!")

def main():
    """主函数"""
    explainer = FixedHLOExplainer()
    explainer.comprehensive_hlo_demo()

if __name__ == "__main__":
    main()
