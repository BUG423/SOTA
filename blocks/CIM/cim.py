import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：CIM (Contextual Information Modulator) —— 上下文信息调制器

一、模块简介
在视觉任务中，不同图像区域需要不同程度的上下文信息。小物体或纹理区域主要依赖
局部细节，而大物体或语义区域需要更大的感受野来捕获全局上下文。现有模块（如 ASPP、
RFB）通常将多尺度特征简单拼接或相加，没有显式考虑每个空间位置对上下文的
差异性需求。

CIM 的核心思想是：学习一个空间位置相关的"上下文混合比例"，在每个位置自适应
地决定应该保留多少局部信息、引入多少全局上下文。这类似于人类在观察场景时的
注意力分配——我们既关注细节，也结合整体语境。

核心创新点：
1. 空间自适应上下文混合：每个位置独立学习局部 vs 全局的混合比例
2. 双路径互补设计：局部路径捕获细节，上下文路径提供全局视野
3. 轻量级混合控制器：通过瓶颈结构高效预测混合权重，计算开销极小
4. 可学习的上下文范围：上下文路径的膨胀率可配置，适应不同任务需求

二、结构设计
CIM 由以下子结构组成：
1. 局部特征提取器（Local Feature Extractor）：
   - 使用 3x3 深度可分离卷积提取局部细节特征
2. 上下文特征提取器（Context Feature Extractor）：
   - 使用膨胀率为 d 的 5x5 深度可分离卷积，在保持分辨率的同时扩大感受野
3. 混合控制器（Mix Controller）：
   - 拼接局部和上下文特征，通过 1x1 卷积预测逐通道的混合比例
   - 使用 Sigmoid 将比例限制在 [0, 1]
4. 混合输出：out = α * local_feat + (1-α) * context_feat + input（残差连接）

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 CIM（Contextual Information Modulator）模块，用于自适应地调制特征
的上下文信息量。该模块通过双路径设计分别提取局部细节和全局上下文特征，并利用
轻量级混合控制器为每个空间位置学习最优的局部-上下文融合比例，从而在不显著增加
计算量的前提下丰富特征的表达能力。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务。特别适合需要同时处理多尺度目标
的场景（如小目标检测、高分辨率语义分割）。可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。
'''


class CIM(nn.Module):
    """CIM: Contextual Information Modulator —— 上下文信息调制器"""

    def __init__(self, channels: int, dilation: int = 3, hidden_ratio: int = 8):
        super().__init__()
        hidden_dim = max(channels // hidden_ratio, 4)

        # 局部特征提取：3x3 深度可分离卷积
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # 上下文特征提取：膨胀深度可分离卷积
        self.context_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # 混合控制器：预测每个位置的 local/context 混合比例
        self.mix_controller = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        local_feat = self.local_conv(x)                           # [B, C, H, W]
        context_feat = self.context_conv(x)                       # [B, C, H, W]

        # 预测混合比例 α ∈ [0, 1]
        mix_input = torch.cat([local_feat, context_feat], dim=1)  # [B, 2C, H, W]
        alpha = self.mix_controller(mix_input)                     # [B, C, H, W]

        # 混合 + 残差
        modulated = alpha * local_feat + (1 - alpha) * context_feat
        out = x + modulated
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CIM(channels=128, dilation=3)

    output = model(input_tensor)

    print('=== CIM: Contextual Information Modulator ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))

    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops)
        print('Params:', params)
    except Exception as e:
        print('FLOPs 统计失败，请确认是否安装 thop:', e)
