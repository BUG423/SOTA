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
    ├── RIM/  递归推理模块
    ├── PDR/  极化双表示模块
    ├── SSM/  显著性引导抑制模块
    ├── PGM/  渐进式门控模块
    ├── FEM/  特征均衡模块
    ├── IGM/  信息汇聚模块
    ├── RGM/  互惠引导模块
    ├── DSM/  双尺度调制器
    ├── OEM/  序统计增强模块
    ├── MPM/  动量传播模块
    ├── PCM/  相位一致性模块
    ├── CGM/  条件门控模块
    ├── IRM/  信息路由模块
    ├── BSM/  双边相似度模块
    ├── DGM/  多样性引导模块
    ├── SUM/  空间不确定性模块
    ├── AGM/  自适应粒度模块
    ├── RAM/  残差放大模块
    └── ...
```

## 已实现模块

| 日期 | 模块 | 全称 | 核心思想 | 适用任务 |
|------|------|------|----------|----------|
| 05-27 | SRM | Selective Response Module | 分组统计→位置敏感通道调制→软阈值稀疏化 | 分类/检测/分割 |
| 05-27 | DFA | Differential Feature Amplifier | 局部邻域差异→差异驱动放大→对比度敏感 | 分类/检测/边缘检测 |
| 05-27 | CIM | Contextual Information Modulator | 双路径(局部+上下文)→空间自适应混合比例 | 分类/检测/分割 |
| 05-27 | GFF | Gated Feature Fusion | 三路并行变换→空间-通道联合门控→竞争性融合 | 分类/检测/分割 |
| 05-27 | DRS | Dynamic Receptive Field Selector | 多膨胀率并行分支→空间自适应感受野选择 | 检测/分割(多尺度) |
| 05-27 | AFM | Adaptive Frequency Modulation | 多核并行→频率带分解→空间自适应频率调制 | 分类/检测/图像恢复 |
| 05-27 | PFA | Progressive Feature Aggregator | 两阶段粗调-精调→阶段间信息桥接→残差累积 | 分类/检测/分割 |
| 05-27 | SAM | Spatial Affinity Module | 低秩投影→亲和力矩阵→信息传播→全局上下文 | 分割/检测/生成 |
| 05-27 | CRM | Channel Recalibration Module | 激活熵估计→熵引导通道评估→冗余抑制 | 分类/检测/分割 |
| 05-27 | LCR | Local Context Reconstructor | 逐位置动态邻域权重→专属局部卷积核→自适应聚合 | 分类/检测/分割 |
| 05-28 | RIM | Recursive Inference Module | 权重共享递归变换→迭代嵌入→残差累积精炼 | 分类/检测/分割 |
| 05-28 | PDR | Polarized Dual Representation | 空间/语义双通路→交叉门控→极化特征融合 | 分类/检测/分割 |
| 05-28 | SSM | Saliency-Guided Suppression Module | 显著性检测→自适应阈值→软抑制→信息预算重分配 | 分类/检测/分割 |
| 05-29 | PGM | Progressive Gating Module | 三阶段级联门控(粗→中→细)→通道级自适应融合 | 分类/检测/分割 |
| 05-29 | FEM | Feature Equilibrium Module | 通道统计编码→均衡能量学习→指数平滑调节 | 分类/检测/分割 |
| 05-29 | IGM | Information Gathering Module | 多尺度深度可分离汇聚→空间自适应尺度权重 | 分类/检测/分割 |
| 05-30 | RGM | Reciprocal Guidance Module | 通道-空间双分支互惠引导→双向信息调制 | 分类/检测/分割 |
| 05-30 | DSM | Dual-Scale Modulator | 粗-细双尺度互调→上下文引导+细节回注 | 分类/检测/分割 |
| 05-30 | OEM | Order-Statistic Enhancement Module | 多序统计量提取→空间自适应统计选择→鲁棒增强 | 分类/检测/分割 |
| 05-31 | MPM | Momentum Propagation Module | 大核平滑动量参考→瞬态偏差感知→自适应调制 | 分类/检测/分割 |
| 05-31 | PCM | Phase-Coherence Module | FFT频域解耦→幅度重标定+相位一致性增强 | 分类/检测/分割 |
| 05-31 | CGM | Conditional Gating Module | 可学习条件原型→相似度驱动门控→语义参照增强 | 分类/检测/分割 |
| 06-01 | IRM | Information Routing Module | 多专家内容感知路由→空间自适应专家混合 | 分类/检测/分割 |
| 06-01 | BSM | Bilateral Similarity Module | K×K邻域内容相似度→双边自适应加权聚合 | 分类/检测/分割 |
| 06-01 | DGM | Diversity-Guided Module | 通道Gram矩阵→冗余分数驱动调制→多样性增强 | 分类/检测/分割 |
| 06-03 | SUM | Spatial Uncertainty Module | 局部不确定性估计→不确定引导平滑/保持双路径融合 | 分类/检测/分割 |
| 06-03 | AGM | Adaptive Granularity Module | 粒度偏好图→粗细双分支→空间自适应粒度插值 | 分类/检测/分割 |
| 06-03 | RAM | Residual Amplification Module | 基座-残差分解→残差信息分析→内容感知放大/抑制 | 分类/检测/分割 |

## 使用方法

每个模块均可作为即插即用组件嵌入现有网络：

```python
from blocks.RIM.rim import RIM
import torch

rim = RIM(channels=64, num_iterations=3)
x = torch.randn(1, 64, 32, 32)
out = rim(x)
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
