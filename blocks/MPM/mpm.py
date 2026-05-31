import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-31

'''
模块名称：MPM (Momentum Propagation Module) —— 动量传播模块

一、模块简介
在深度卷积神经网络中，特征图在不同层之间传递时，噪声和伪影会逐层累积，
导致深层特征的信噪比下降。虽然 BatchNorm 和 LayerNorm 等归一化方法能够
稳定训练过程中的激活分布，但它们通常只对单层特征进行全局或局部标准化，
缺乏对特征在"时间维度"（层间传递方向）上演变趋势的建模能力。此外，
传统卷积产生的逐层特征变化缺乏内在的正则化机制，对输入扰动较为敏感。

MPM 的核心思想是：借鉴动量（momentum）在优化和时序平滑中的思想，为特征
层间传播引入动量传播机制。具体而言，MPM 对输入特征进行大核平滑以构建
"动量参考"——代表特征的稳定趋势分量；随后计算原始特征与动量参考之间的
偏差（即瞬态波动分量）；再通过一个可学习的偏差调制网络，根据偏差幅度
自适应地调整特征响应——在偏差较大的区域（可能为噪声或伪影）进行软抑制，
在偏差较小的区域（稳定特征）保持原有响应。这种"以趋势为锚、抑制瞬态
波动"的机制为特征传播提供了内建的正则化效果。

核心创新点：
1. 动量参考构建：通过大核平滑提取特征的稳定趋势分量，作为调制参考
2. 瞬态偏差感知：计算原始特征与动量参考的偏差，量化各位置的波动程度
3. 自适应偏差调制：通过偏差驱动的调制网络，对不同波动程度的区域进行差异化处理
4. 平滑先验注入：将动量参考与调制后的瞬态分量重新组合，实现信息保真

二、结构设计
MPM 由以下子结构组成：
1. 动量提取器（Momentum Extractor）：
   - 5x5 逐通道卷积（depthwise）+ BatchNorm
   - 通过大感受野平滑操作提取特征的稳定趋势分量
2. 偏差计算（Deviation）：
   - 原始特征 - 动量参考，得到瞬态波动分量
3. 偏差调制网络（Deviation Modulation Network）：
   - 1x1 卷积降维 → GELU → 1x1 卷积 → Sigmoid
   - 以偏差幅度为输入，输出 [0,1] 范围的调制权重
4. 自适应重组：
   - 调制后的瞬态分量与动量参考重新组合
   - 通过可学习权重平衡两者贡献
   - 残差连接输出

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 MPM（Momentum Propagation Module）模块，通过引入动量传播机制
为特征层间传递提供内建正则化。该模块首先通过大核平滑构建特征的动量参考，
将其作为特征的稳定趋势估计；随后计算原始特征相对于动量参考的瞬态偏差，
并通过偏差驱动的自适应调制网络对不同波动程度的区域进行差异化处理，
从而在保留有效特征响应的同时抑制噪声累积。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合网络层数较深、特征传递路径较长的
场景，以及需要提升模型抗噪能力的任务。
'''


class MPM(nn.Module):
    """MPM: Momentum Propagation Module —— 动量传播模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Momentum extractor: large-kernel smoothing to capture stable trends
        self.momentum_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                      groups=channels, bias=False)
        self.momentum_bn = nn.BatchNorm2d(channels)

        # Deviation modulation network
        self.deviation_mod = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Learnable per-channel momentum-vs-detail balance
        self.momentum_weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 1. Extract momentum reference: large-kernel smoothing
        momentum = self.momentum_dw(x)
        momentum = self.momentum_bn(momentum)
        momentum = F.gelu(momentum)                                  # [B, C, H, W]

        # 2. Compute deviation: transient fluctuation component
        deviation = x - momentum                                      # [B, C, H, W]

        # 3. Deviation-driven modulation: suppress where deviation is large
        mod = self.deviation_mod(deviation)                           # [B, C, H, W]
        detail_modulated = deviation * mod                            # suppressed deviation

        # 4. Adaptive recombination: balance momentum and modulated detail
        # momentum_weight in [0, 1] via sigmoid, controls how much momentum to keep
        alpha = self.momentum_weight.sigmoid()                        # [1, C, 1, 1]
        combined = alpha * momentum + (1 - alpha) * detail_modulated  # [B, C, H, W]

        # 5. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = MPM(channels=128)
    output = model(input_tensor)
    print('=== MPM: Momentum Propagation Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('momentum_weight (sigmoid):', model.momentum_weight.sigmoid().mean().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
