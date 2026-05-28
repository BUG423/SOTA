import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-28

'''
模块名称：RIM (Recursive Inference Module) —— 递归推理模块

一、模块简介
大多数特征增强模块（如 SE、CBAM、SRM）采用单次前向计算的方式：输入特征经过
一组固定的变换后输出增强特征。这种"一步到位"的方式虽然高效，但缺乏对特征的
反复雕琢能力。在传统的优化理论中，许多问题通过迭代算法（如梯度下降）逐步逼近
最优解；而在神经网络中，深度本身就提供了一种逐层迭代的机制，但每层使用不同的
参数，导致参数量线性增长。

RIM 的核心思想是：将特征增强建模为递归推理过程。同一个轻量级变换块被重复应用
多次（如 3 次），每次接收上一次的输出作为输入，逐步精炼特征。这种设计在保持
极少参数量的同时（因为权重共享），赋予了模块"反复思考"的能力。

核心创新点：
1. 权重共享的递归推理：同一个变换块应用多次，参数量不随迭代次数增长
2. 迭代嵌入（Iteration Embedding）：为每次迭代学习一个独特的偏置，使共享的
   变换块能感知当前迭代的阶段，产生差异化行为
3. 残差累积路径：每次迭代的增强信号以残差方式累积，确保信息不丢失
4. 自适应迭代深度：理论上可在推理时动态调整迭代次数以权衡精度与速度

二、结构设计
RIM 由以下子结构组成：
1. 共享变换块（Shared Transformation Block）：
   - 3x3 深度可分离卷积 + 1x1 通道混合
   - 所有迭代共享此块的参数
2. 迭代嵌入（Iteration Embedding）：
   - 为每次迭代学习一个 C 维向量和 H×W 维的空间偏置
   - 在每次迭代开始时加到特征上，使共享块能区分不同迭代阶段
3. 残差累积器：
   - 每次迭代的输出通过可学习的权重累加到原始输入上
   - 最终输出 = x + Σ w_i * output_i

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 RIM（Recursive Inference Module）模块，通过权重共享的递归推理机制
实现参数高效的特征增强。该模块将同一个轻量级变换块重复应用多次，并引入迭代
嵌入使共享块在不同迭代阶段产生差异化行为，从而以极小的参数量获得迭代精炼的
收益，类似于计算机视觉中的递归推理过程。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。特别适合对参数量敏感的场景（如移动端部署），因为权重
共享使参数量不随迭代次数增加。
'''


class RIM(nn.Module):
    """RIM: Recursive Inference Module —— 递归推理模块"""

    def __init__(self, channels: int, num_iterations: int = 3):
        super().__init__()
        self.num_iterations = num_iterations

        # 共享变换块
        self.shared_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # 迭代嵌入（每一步有不同的偏置）
        self.iter_embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(1, channels, 1, 1))
            for _ in range(num_iterations)
        ])

        # 每次迭代的累积权重（可学习）
        self.iter_weights = nn.Parameter(torch.ones(num_iterations) / num_iterations)

        # 最终输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        accumulated = torch.zeros_like(x)
        h = x

        for t in range(self.num_iterations):
            # 注入迭代嵌入
            h = h + self.iter_embeddings[t]
            # 共享变换
            h = self.shared_block(h)
            # 残差累积
            accumulated = accumulated + self.iter_weights[t] * h

        out = self.output_proj(accumulated)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = RIM(channels=128, num_iterations=3)
    output = model(input_tensor)
    print('=== RIM: Recursive Inference Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('iter_weights:', model.iter_weights.data)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
