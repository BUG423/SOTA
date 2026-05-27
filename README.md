# SOTA - State-of-the-Art Neural Network Modules

一个系统整理、复现和开发常用神经网络模块的开源仓库，重点面向计算机视觉任务。

## 项目简介

本项目收集和实现了计算机视觉领域常见的即插即用神经网络模块，涵盖图像分类、目标检测、语义分割等方向。每个模块都包含清晰的论文背景说明、结构设计说明、PyTorch 代码实现和测试代码。**所有模块均为原创设计。**

## 目录结构

```
SOTA/
├── README.md
├── resnet_insert_example.py
└── blocks/
    ├── SRM/  选择性响应模块
    ├── DFA/  差异性特征放大器
    ├── CIM/  上下文信息调制器
    ├── GFF/  门控特征融合模块
    ├── DRS/  动态感受野选择器
    ├── AFM/  自适应频率调制模块
    ├── PFA/  渐进式特征聚合器
    ├── SAM/  空间亲和力模块
    ├── CRM/  通道重校准模块
    ├── LCR/  局部上下文重构模块
    └── ...
```

## 已实现模块

| 模块 | 全称 | 核心思想 | 适用任务 |
|------|------|----------|----------|
| SRM | Selective Response Module | 分组统计→位置敏感通道调制→软阈值稀疏化 | 分类/检测/分割 |
| DFA | Differential Feature Amplifier | 局部邻域差异→差异驱动放大→对比度敏感 | 分类/检测/边缘检测 |
| CIM | Contextual Information Modulator | 双路径(局部+上下文)→空间自适应混合比例 | 分类/检测/分割 |
| GFF | Gated Feature Fusion | 三路并行变换→空间-通道联合门控→竞争性融合 | 分类/检测/分割 |
| DRS | Dynamic Receptive Field Selector | 多膨胀率并行分支→空间自适应感受野选择 | 检测/分割(多尺度) |
| AFM | Adaptive Frequency Modulation | 多核并行→频率带分解→空间自适应频率调制 | 分类/检测/图像恢复 |
| PFA | Progressive Feature Aggregator | 两阶段粗调-精调→阶段间信息桥接→残差累积 | 分类/检测/分割 |
| SAM | Spatial Affinity Module | 低秩投影→亲和力矩阵→信息传播→全局上下文 | 分割/检测/生成 |
| CRM | Channel Recalibration Module | 激活熵估计→熵引导通道评估→冗余抑制 | 分类/检测/分割 |
| LCR | Local Context Reconstructor | 逐位置动态邻域权重→专属局部卷积核→自适应聚合 | 分类/检测/分割 |

## 使用方法

每个模块均可作为即插即用组件嵌入现有网络：

```python
from blocks.SRM.srm import SRM
import torch

srm = SRM(channels=64)
x = torch.randn(1, 64, 32, 32)
out = srm(x)
print(out.shape)  # [1, 64, 32, 32]
```

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.10
- thop（可选，用于 FLOPs 统计）

## 贡献指南

欢迎提交 PR 添加新的神经网络模块。请参考 `blocks/` 下已有模块的格式，确保包含完整的文档说明和测试代码。

## 许可证

MIT License
