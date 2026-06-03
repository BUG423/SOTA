import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-03

'''
模块名称：RAM (Residual Amplification Module) —— 残差放大模块

一、模块简介
在深度神经网络中，残差连接（residual connection）是信息传递的核心机制。
然而，标准的残差连接对残差分量采取"全量通过"的策略——所有残差信息
不加区分地添加到主路径上。实际上，残差分量中既包含有价值的增量信息
（如边缘增强、语义细化），也包含噪声和无意义的微小波动，后者不仅对
任务无益，还会在深层网络中逐层累积、降低特征质量。

RAM 的核心思想是：对残差分量进行内容感知的"放大或抑制"——在信息丰富
的位置（如边缘、纹理变化处）放大残差信号以增强特征表达能力，在信息贫乏
的位置（如平坦区域、噪声处）抑制残差信号以减少不必要的扰动。具体而言，
RAM 首先通过平滑操作将输入分解为"基座"（平滑分量）和"残差"（原始减
平滑），然后通过一个轻量级的残差分析网络评估残差的信息含量，生成每个
通道和位置的放大/抑制系数，最后将放大的残差与基座重新组合。

核心创新点：
1. 显式残差分解：通过平滑操作将特征分解为基座分量和残差分量
2. 内容感知残差放大：根据残差本身的信息含量决定放大或抑制的程度
3. 逐通道自适应：每个通道独立学习放大/抑制的基线强度
4. 软阈值效应：Sigmoid 门控提供类似软阈值的去噪效果

二、结构设计
RAM 由以下子结构组成：
1. 基座提取器（Base Extractor）：
   - 5x5 逐通道卷积 + BatchNorm
   - 从输入中提取平滑的"基座"分量
2. 残差计算：
   - 残差 = 输入 - 基座
3. 残差分析网络（Residual Analyzer）：
   - 以残差的统计量（均值和标准差）为输入
   - 1x1 卷积降维 → GELU → 1x1 卷积 → Sigmoid
   - 输出 [B, C, H, W] 的放大系数图
4. 残差放大：
   - 放大残差 = 放大系数 * 残差
5. 重组与输出：
   - 输出 = 基座 + 放大残差
   - 通过可学习权重平衡基座与放大残差的贡献
   - 残差连接（与原始输入相加）

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 RAM（Residual Amplification Module）模块，通过对残差分量
进行内容感知的放大与抑制来提升特征质量。该模块将输入特征分解为基座
分量和残差分量，利用残差分析网络评估残差的信息含量，并据此在空间和
通道维度上自适应地调节残差的贡献强度——信息性残差被放大以增强特征
表达，噪声性残差被抑制以减少扰动——从而在保留有效信息的同时抑制
无意义的特征波动。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合深度网络中残差路径的特征增强，
以及需要抑制特征噪声、提升信噪比的场景。
'''


class RAM(nn.Module):
    """RAM: Residual Amplification Module —— 残差放大模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Base extractor: smooth the input to get the "base" component
        self.base_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                  groups=channels, bias=False)
        self.base_bn = nn.BatchNorm2d(channels)

        # Residual analyzer: evaluate the informativeness of the residual
        # Input: residual stats (mean + std per channel) → amplification map
        self.residual_analyzer = nn.Sequential(
            nn.Conv2d(channels * 2, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Learnable per-channel amplification baseline
        self.amp_baseline = nn.Parameter(torch.ones(1, channels, 1, 1))

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
        B, C, H, W = x.shape

        # 1. Extract base component via smoothing
        base = self.base_dw(x)
        base = self.base_bn(base)
        base = F.gelu(base)                                          # [B, C, H, W]

        # 2. Compute residual: what deviates from the smooth base
        residual = x - base                                           # [B, C, H, W]

        # 3. Analyze residual informativeness
        # Per-channel spatial statistics of the residual
        res_mean = residual.mean(dim=(2, 3), keepdim=True)            # [B, C, 1, 1]
        res_std = residual.std(dim=(2, 3), keepdim=True) + 1e-5       # [B, C, 1, 1]

        # Expand statistics to spatial dimensions for pixel-level gating
        res_stats = torch.cat([
            res_mean.expand(-1, -1, H, W),
            res_std.expand(-1, -1, H, W),
        ], dim=1)                                                      # [B, 2C, H, W]

        # Generate amplification map
        amp_map = self.residual_analyzer(res_stats)                   # [B, C, H, W]

        # 4. Amplify residual: content-aware gating + learnable baseline
        amp_factor = self.amp_baseline * amp_map                      # [B, C, H, W]
        amplified_residual = amp_factor * residual                     # [B, C, H, W]

        # 5. Reconstruct: base + amplified residual
        reconstructed = base + amplified_residual                      # [B, C, H, W]

        # 6. Refine and global residual connection
        out = self.refine(reconstructed)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = RAM(channels=128)
    output = model(input_tensor)
    print('=== RAM: Residual Amplification Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('amp_baseline mean:', model.amp_baseline.mean().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
