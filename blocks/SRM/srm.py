import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：SRM (Selective Response Module) —— 选择性响应模块

一、模块简介
现有注意力机制（如 SE、CBAM）通常对所有空间位置施加统一的通道注意力或对所有
通道施加统一的空间注意力。然而，不同空间位置的特征对不同通道的依赖程度并不相同。
例如，边缘区域更依赖纹理相关的通道，而平坦区域更依赖颜色相关的通道。

SRM 提出了一种"选择性响应"机制：对每个空间位置，根据其局部统计特性（均值、
标准差），自适应地选择增强哪些通道、抑制哪些通道。与 SE 的全局通道注意力不同，
SRM 生成的是空间位置相关的通道响应权重，实现更细粒度的特征调制。

核心创新点：
1. 位置敏感的通道调制：每个空间位置有其独立的通道重要性权重
2. 统计驱动的门控：使用局部统计量（均值、方差）作为调制信号的来源，
   比直接使用原始特征更稳定
3. 软阈值去噪：通过可学习的软阈值机制，自动抑制低响应特征

二、结构设计
SRM 由以下子结构组成：
1. 局部分组统计提取器（Local Group Statistics Extractor）：
   - 将通道分为 G 组，每组计算每个空间位置的均值和标准差
   - 输出形状为 [B, 2G, H, W] 的统计特征图
2. 门控权重生成器（Gate Weight Generator）：
   - 对统计特征使用 1x1 卷积和激活函数，生成 [B, C, H, W] 的门控权重
   - 使用 Tanh 激活使权重范围为 [-1, 1]，允许增强或抑制
3. 软阈值单元（Soft Threshold Unit）：
   - 学习一个全局阈值参数 τ
   - 对权重应用软阈值：sign(w) * max(|w| - τ, 0)
   - 将接近零的权重压制为零，实现稀疏激活
4. 残差连接：最终输出 = 输入 + 调制后的输入

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 SRM（Selective Response Module）模块，用于增强特征的选择性表达能力。
该模块通过提取每个空间位置的分组统计信息，生成位置敏感的通道调制权重，并引入
软阈值机制进行稀疏化，从而在不显著增加计算量的前提下提升特征的判别能力。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络的任意中间层。
'''


class SRM(nn.Module):
    """SRM: Selective Response Module —— 选择性响应模块"""

    def __init__(self, channels: int, groups: int = 8, threshold_init: float = 0.1):
        super().__init__()
        assert channels % groups == 0, f"channels ({channels}) must be divisible by groups ({groups})"
        self.channels = channels
        self.groups = groups
        self.ch_per_group = channels // groups

        # 门控网络：从 2*G 维统计量映射到 C 维门控权重
        self.gate_generator = nn.Sequential(
            nn.Conv2d(groups * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Tanh(),
        )

        # 可学习的软阈值参数
        self.threshold = nn.Parameter(torch.tensor(threshold_init))

    def _compute_group_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算每个通道组在每个空间位置的均值和标准差。
        x: [B, C, H, W] -> stats: [B, 2*G, H, W]
        """
        B, C, H, W = x.shape
        x_grouped = x.view(B, self.groups, self.ch_per_group, H, W)

        mean = x_grouped.mean(dim=2)     # [B, G, H, W]
        std = x_grouped.std(dim=2)        # [B, G, H, W]

        stats = torch.cat([mean, std], dim=1)  # [B, 2G, H, W]
        return stats

    @staticmethod
    def _soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """软阈值函数：sign(x) * max(|x| - τ, 0)"""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        stats = self._compute_group_statistics(x)           # [B, 2G, H, W]
        gate_weights = self.gate_generator(stats)            # [B, C, H, W]
        gate_weights = self._soft_threshold(gate_weights, self.threshold)
        out = x + x * gate_weights
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SRM(channels=128, groups=8)

    output = model(input_tensor)

    print('=== SRM: Selective Response Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('threshold:', model.threshold.item())

    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops)
        print('Params:', params)
    except Exception as e:
        print('FLOPs 统计失败，请确认是否安装 thop:', e)
