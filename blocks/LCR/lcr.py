import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：LCR (Local Context Reconstructor) —— 局部上下文重构模块

一、模块简介
标准卷积使用固定的、所有空间位置共享的卷积核来聚合邻域信息。这种"一刀切"的
聚合方式忽视了不同空间位置对邻域信息的差异化需求。例如，边缘位置可能需要
沿边缘方向的邻域信息，而角点位置可能需要来自多个方向的信息。

LCR 的核心思想是：为每个空间位置动态预测一个局部的重构权重矩阵，用该权重矩阵
对其 k×k 邻域内的特征进行加权组合，从而"重构"出该位置的新特征表示。这相当于
为每个像素学习一个专属的小型动态卷积核。

核心创新点：
1. 逐位置动态邻域权重：每个空间位置独立学习其 k×k 邻域内的聚合权重
2. 轻量级权重预测器：使用深度可分离卷积 + 通道压缩来预测权重，计算量极小
3. 保持平移等变性：权重基于局部特征内容预测，保持空间结构
4. 显式的局部结构建模：相比全局注意力，LCR 更专注于高效地建模局部上下文

二、结构设计
LCR 由以下子结构组成：
1. 权重预测器（Weight Predictor）：
   - 通过深度可分离卷积和 1x1 卷积预测每个位置的 k² 个邻域权重
   - 使用 softmax 归一化使权重和为 1
2. 邻域展开器（Neighborhood Unfolder）：
   - 使用 nn.Unfold 将输入展开为 [B, C×k², HW] 的形式
   - 每个位置对应 k×k 邻域内的所有特征值
3. 动态重构：
   - 将预测的权重与展开的邻域特征进行逐位置加权求和
   - 权重在通道维度上共享（每组通道使用相同的空间权重）
4. 残差输出：out = x + reconstructed

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 LCR（Local Context Reconstructor）模块，通过动态预测逐位置的局部
重构权重来实现自适应的邻域信息聚合。该模块为每个空间位置学习其专属的 k×k
邻域聚合模式，克服了标准卷积使用固定共享核的局限性，在不显著增加计算量的
前提下提升了局部上下文建模的灵活性。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。对需要精细局部结构建模的下游任务尤为有效。
'''


class LCR(nn.Module):
    """LCR: Local Context Reconstructor —— 局部上下文重构模块"""

    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 4):
        super().__init__()
        assert channels % groups == 0, f"channels must be divisible by groups"
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.groups = groups
        self.ch_per_group = channels // groups

        # 权重预测器
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, groups * self.k2, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        pad = self.kernel_size // 2

        # 预测每个位置的邻域权重
        weights = self.weight_predictor(x)                        # [B, G*k², H, W]
        weights = weights.reshape(B, self.groups, self.k2, H * W) # [B, G, k², HW]
        weights = torch.softmax(weights, dim=2)                   # 归一化

        # 展开邻域特征
        unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=pad)  # [B, C*k², HW]

        # 按组进行动态重构
        out_parts = []
        for g in range(self.groups):
            c_start = g * self.ch_per_group
            c_end = (g + 1) * self.ch_per_group
            # 当前组的邻域特征 [B, C/G * k², HW]
            feat_g = unfolded[:, c_start * self.k2:c_end * self.k2, :]
            feat_g = feat_g.reshape(B, self.ch_per_group, self.k2, H * W)  # [B, C/G, k², HW]
            # 当前组的权重 [B, 1, k², HW]
            w_g = weights[:, g:g + 1, :, :]
            # 加权求和
            out_g = (feat_g * w_g).sum(dim=2)                    # [B, C/G, HW]
            out_parts.append(out_g)

        out = torch.cat(out_parts, dim=1)                         # [B, C, HW]
        out = out.reshape(B, C, H, W)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = LCR(channels=128, kernel_size=3, groups=4)
    output = model(input_tensor)
    print('=== LCR: Local Context Reconstructor ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
