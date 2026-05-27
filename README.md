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
    ├── SCSA/
    │   └── scsa.py
    ├── EMA/
    │   └── ema.py
    ├── CBAM/
    │   └── cbam.py
    └── ...
```

## 已实现模块

| 模块 | 论文 | 适用任务 |
|------|------|----------|
| CBAM | CBAM: Convolutional Block Attention Module (ECCV 2018) | 图像分类、目标检测 |
| EMA | Efficient Multi-Scale Attention (ICASSP 2023) | 图像分类、目标检测、语义分割 |
| SCSA | Spatial and Channel Self-Attention | 图像分类、目标检测、语义分割 |

## 使用方法

每个模块均可作为即插即用组件嵌入现有网络：

```python
from blocks.CBAM.cbam import CBAM
import torch

cbam = CBAM(channels=64)
x = torch.randn(1, 64, 32, 32)
out = cbam(x)
print(out.shape)  # [1, 64, 32, 32]
```

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.10
- einops
- thop（可选，用于 FLOPs 统计）

## 贡献指南

欢迎提交 PR 添加新的神经网络模块。请参考 `blocks/` 下已有模块的格式，确保包含完整的文档说明和测试代码。

## 许可证

MIT License
