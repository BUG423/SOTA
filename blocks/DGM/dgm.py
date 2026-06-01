import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-01

'''
模块名称：DGM (Diversity-Guided Module) —— 多样性引导模块

一、模块简介
在深度卷积神经网络中，随着层数的增加，通道数量通常被设计为逐步扩大，
以增加特征表示的容量。然而，更大的通道数并不自动意味着更丰富的表示——
网络在训练过程中可能学到冗余的通道，即多个通道提取高度相似的特征模式，
导致有效特征维度远小于通道数。这种现象被称为"特征坍塌"（feature
collapse），其后果是浪费了宝贵的模型容量和计算资源，却没有带来相应的
表示能力提升。

DGM 的核心思想是：显式地评估通道间的特征多样性，并以多样性为引导信号
对通道响应进行调制——鼓励独特通道、抑制冗余通道。具体而言，DGM 计算
通道特征之间的 Gram 矩阵来衡量通道间相似度，每个通道与所有其他通道的
累积相似度构成其"冗余分数"。冗余分数高的通道（与其他通道过于相似）
被软抑制，冗余分数低的通道（独特的、不可替代的）被保留甚至增强。
这种多样性引导的调制机制作为梯度信号的一个补充偏置，促使网络学习
更去相关化、更互补的通道特征模式。

核心创新点：
1. 通道多样性度量：通过 Gram 矩阵量化通道间的特征冗余程度
2. 冗余自适应调制：以冗余分数驱动通道级别的软调制，抑制冗余增强多样性
3. 高效实现：Gram 矩阵计算仅需 O(C²·H·W)，对常用通道数完全可接受
4. 即插即用：无需修改损失函数或训练流程，作为无监督的正则化模块运作

二、结构设计
DGM 由以下子结构组成：
1. 特征归一化：
   - 对每个通道的特征进行 L2 归一化（沿空间维度）
   - 使相似度计算专注于方向而非幅值
2. Gram 矩阵计算：
   - 归一化特征：f ∈ [B, C, N]（N = H*W）
   - Gram = f @ f.T ∈ [B, C, C]
   - Gram[i,j] 度量通道 i 与通道 j 的相似度
3. 冗余分数提取：
   - 每个通道的冗余分数 = sum(Gram[i, :]) - 1.0（排除了自相似度 Gram[i,i]）
   - 冗余分数高 → 该通道与很多其他通道相似 → 应被抑制
4. 冗余-调制映射：
   - 将冗余分数通过一个小型全连接网络映射为调制权重
   - 高冗余 → 低权重（抑制），低冗余 → 高权重（增强）
5. 调制应用与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 DGM（Diversity-Guided Module）模块，通过显式建模通道间特征
多样性来引导自适应特征调制。该模块计算通道 Gram 矩阵以量化通道间的特征
冗余程度，并基于每个通道的冗余分数生成调制信号——高冗余通道被软抑制、
低冗余通道被保留增强——从而在不引入额外监督信号的前提下促进网络学习
去相关化、多样化的通道表示。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合通道数较大、容易产生通道冗余的
深层网络，以及需要在有限参数量下提高特征利用效率的轻量化模型。
'''


class DGM(nn.Module):
    """DGM: Diversity-Guided Module —— 多样性引导模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Redundancy-to-modulation mapping
        # Input: per-channel redundancy score [B, C]
        # Output: per-channel modulation weight [B, C]
        self.modulation = nn.Sequential(
            nn.Linear(channels, inner),
            nn.GELU(),
            nn.Linear(inner, channels),
            nn.Sigmoid(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Learnable modulation strength
        self.strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # 1. Reshape and L2-normalize each channel (along spatial dim)
        feat = x.view(B, C, N)                                       # [B, C, N]
        feat_norm = F.normalize(feat, p=2, dim=-1)                   # [B, C, N]

        # 2. Compute Gram matrix: G[b, i, j] = similarity(ch_i, ch_j)
        G = torch.bmm(feat_norm, feat_norm.transpose(1, 2))          # [B, C, C]

        # 3. Per-channel redundancy score: sum of similarities to all other channels
        # G.sum(dim=-1) = row sum, subtract 1.0 to remove self-similarity
        redundancy = G.sum(dim=-1) - 1.0                              # [B, C]
        # Normalize by (C-1) to get average similarity per channel
        redundancy = redundancy / max(1, C - 1)                      # [B, C]

        # 4. Map redundancy to modulation weight
        # High redundancy → lower weight (suppress), low redundancy → higher weight
        mod_weight = self.modulation(redundancy)                      # [B, C]
        # Invert: high redundancy should be suppressed
        mod_weight = 1.0 - self.strength.sigmoid() * mod_weight      # [B, C]
        mod_weight = mod_weight.view(B, C, 1, 1)                      # [B, C, 1, 1]

        # 5. Apply modulation and residual
        out = x * mod_weight
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = DGM(channels=128)
    output = model(input_tensor)
    print('=== DGM: Diversity-Guided Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('modulation_strength:', model.strength.sigmoid().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
