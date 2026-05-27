# SOTA - State-of-the-Art Neural Network Modules

一个系统整理、复现和开发常用神经网络模块的开源仓库，重点面向计算机视觉任务。

## 项目简介

本项目收集和实现了计算机视觉领域常见的即插即用神经网络模块，涵盖图像分类、目标检测、语义分割等方向。每个模块都包含清晰的论文背景说明、结构设计说明、PyTorch 代码实现和测试代码。

## 目录结构

```
SOTA/
├── README.md
├── resnet_insert_example.py
└── blocks/
    ├── SRM/
    │   └── srm.py
    ├── DFA/
    │   └── dfa.py
    ├── CIM/
    │   └── cim.py
    └── ...
```

## 已实现模块

| 模块 | 全称 | 类型 | 适用任务 |
|------|------|------|----------|
| SRM | Selective Response Module | 选择性响应注意力 | 图像分类、目标检测、语义分割 |
| DFA | Differential Feature Amplifier | 差异性特征放大 | 图像分类、目标检测、边缘检测 |
| CIM | Contextual Information Modulator | 上下文信息调制 | 图像分类、目标检测、语义分割 |

### SRM (Selective Response Module)

选择性响应模块。通过提取每个空间位置的分组统计信息，生成位置敏感的通道调制权重，并引入软阈值机制进行稀疏化。与 SE 的全局通道注意力不同，SRM 实现了空间位置相关的细粒度特征调制。

### DFA (Differential Feature Amplifier)

差异性特征放大器。显式计算每个位置与其局部邻域的特征差异，差异越大的区域（如边缘、纹理）获得越强的特征增强，模拟人类视觉系统中的对比度敏感机制。

### CIM (Contextual Information Modulator)

上下文信息调制器。通过双路径设计分别提取局部细节和全局上下文特征，并利用轻量级混合控制器为每个空间位置学习最优的融合比例，实现空间自适应的上下文调制。

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
