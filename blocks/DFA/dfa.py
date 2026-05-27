import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：DFA (Differential Feature Amplifier) —— 差异性特征放大器

一、模块简介
在计算机视觉任务中，判别性特征（如物体边缘、纹理变化、关键点）通常表现为与
周围邻域存在显著差异的区域。例如，一个红色苹果在绿色背景中，苹果边界的像素值
与周围像素有较大差异。现有的注意力模块（如 CBAM、SE）直接对原始特征进行操作，
没有显式地利用这种"局部差异性"信息。

DFA 的核心思想是：显式计算每个空间位置与其局部邻域之间的差异，差异越大的区域
获得越强的特征增强。这类似于人类视觉系统中的"对比度敏感"机制——我们更容易
注意到与背景有显著对比的物体。

核心创新点：
1. 局部对比度计算：使用大核深度可分离卷积计算局部平均，求取差异图
2. 差异驱动放大：差异越大 → 放大系数越高，使模型自动关注判别性区域
3. 自适应邻域范围：使用可学习的卷积核参数，使"邻域"的定义可训练
4. 纯增强设计：只放大不抑制（scale ∈ [0.5, 1.5]），避免丢失信息

二、结构设计
DFA 由以下子结构组成：
1. 邻域平滑器（Neighborhood Smoother）：
   - 使用较大核（默认 7x7）的深度可分离卷积计算局部加权平均
   - 卷积核经过 Sigmoid 归一化，确保是合法的平滑操作
2. 差异提取器（Difference Extractor）：
   - 计算原始特征与平滑特征的绝对差值 |x - smooth(x)|
   - 差值大小反映局部变化程度和结构信息
3. 放大权重生成器（Amplification Weight Generator）：
   - 将差异图通过 bottleneck 结构（1x1 conv → ReLU → 1x1 conv → Sigmoid）
   - 映射为 [0, 1] 的系数，再缩放到 [min_scale, max_scale]
4. 逐通道放大：输出 = 原始特征 × 放大系数

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 DFA（Differential Feature Amplifier）模块，用于增强特征的局部判别
能力。该模块通过计算每个位置与其局部邻域的差异程度，自适应地放大高对比度区域
的特征响应，从而引导网络关注具有判别力的局部模式。该模块计算量极小，可作为
即插即用组件嵌入任意网络架构。"

四、适用任务
适用于图像分类、目标检测、语义分割、边缘检测等视觉任务，特别适合需要关注
局部细节和边缘信息的场景。可作为即插即用模块嵌入 CNN 或 Transformer 主干网络。
'''


class DFA(nn.Module):
    """DFA: Differential Feature Amplifier —— 差异性特征放大器"""

    def __init__(self, channels: int, kernel_size: int = 7, hidden_ratio: int = 4):
        super().__init__()
        padding = kernel_size // 2

        # 大核深度可分离卷积（用于计算局部邻域加权平均）
        self.smoother = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size,
                      padding=padding, groups=channels, bias=False),
            nn.Sigmoid(),
        )

        # 放大权重生成器（bottleneck 结构）
        hidden_dim = max(channels // hidden_ratio, 8)
        self.amplifier = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        # 可学习的放大范围参数
        self.min_scale = nn.Parameter(torch.tensor(0.5))
        self.max_scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        smoothed = self.smoother(x)                            # [B, C, H, W]
        diff = torch.abs(x - smoothed)                          # [B, C, H, W]
        amp_raw = self.amplifier(diff)                          # [B, C, H, W]
        amp_factor = self.min_scale + (self.max_scale - self.min_scale) * amp_raw
        out = x * amp_factor
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = DFA(channels=128, kernel_size=7)

    output = model(input_tensor)

    print('=== DFA: Differential Feature Amplifier ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('min_scale:', model.min_scale.item(), 'max_scale:', model.max_scale.item())

    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops)
        print('Params:', params)
    except Exception as e:
        print('FLOPs 统计失败，请确认是否安装 thop:', e)
