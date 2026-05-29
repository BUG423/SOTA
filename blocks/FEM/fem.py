import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-29

'''
模块名称：FEM (Feature Equilibrium Module) —— 特征均衡模块

一、模块简介
在深度卷积神经网络中，不同通道的特征激活幅度往往存在显著差异。部分通道
持续输出高幅值响应，主导了后续层的计算；而另一些通道则长期处于低激活
状态，对网络输出的贡献微乎其微。这种通道间激活能量的不均衡会导致两个问题：
(1) 弱通道的梯度信号被淹没，难以有效训练；(2) 强通道形成"信息瓶颈"，
限制了特征表示的多样性。

FEM 的核心思想是：为每个通道学习一个"均衡能量水平"，并以此为目标对通道
激活进行自适应调节——抑制过度活跃的通道、增强不够活跃的通道，使所有通道
维持在相对均衡的能量水平上。与 SE 等通道注意力模块不同，FEM 不仅评估
通道的重要性，更关注通道间能量分布的平衡性，从而提升特征表示的利用率。

核心创新点：
1. 通道均衡建模：显式学习每个通道的目标能量水平，作为特征调节的参考基线
2. 双统计量联合编码：同时利用通道均值和标准差评估通道当前状态
3. 自适应均衡强度：通过可学习参数控制均衡调节的力度，避免过度干预
4. 软均衡策略：采用平滑的指数调节函数，温和推送而非强制拉平

二、结构设计
FEM 由以下子结构组成：
1. 通道统计提取器：
   - 计算每个通道的均值和标准差（空间维度聚合）
   - 拼接得到 [B, 2C, 1, 1] 的统计描述向量
2. 统计编码网络（Encoder）：
   - 两层 1x1 卷积对统计向量进行非线性编码
   - 输出 [B, C, 1, 1] 的通道注意力权重
3. 能量均衡器（Equilibrator）：
   - 计算每个通道当前的 RMS 能量
   - 与可学习的均衡目标比较，生成指数形式的均衡增益
   - 均衡增益范围 [exp(-1), exp(1)]，中心值为 1
4. 残差连接：将均衡调制后的特征与原始输入相加

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 FEM（Feature Equilibrium Module）模块，通过显式建模通道间能量
均衡来提升特征表示的质量。该模块为每个通道学习一个均衡目标能量水平，
利用通道统计编码网络评估当前状态，并通过指数形式的平滑调节函数将各通道
推向均衡状态，在不过度干预特征语义的前提下改善特征利用效率。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合网络层数较深、通道数较大的场景，
以及存在通道冗余问题的轻量化网络。
'''


class FEM(nn.Module):
    """FEM: Feature Equilibrium Module —— 特征均衡模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Channel statistics encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 2, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Learnable per-channel equilibrium energy (RMS target)
        self.equilibrium = nn.Parameter(torch.ones(1, channels, 1, 1))

        # Learnable equilibrium strength
        self.strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Per-channel statistics
        ch_mean = x.mean(dim=(2, 3), keepdim=True)             # [B, C, 1, 1]
        ch_std = x.std(dim=(2, 3), keepdim=True) + 1e-5        # [B, C, 1, 1]
        stats = torch.cat([ch_mean, ch_std], dim=1)             # [B, 2C, 1, 1]

        # Channel attention from statistics
        attn = self.encoder(stats)                               # [B, C, 1, 1]

        # Per-channel RMS energy
        energy = x.square().mean(dim=(2, 3), keepdim=True).sqrt() + 1e-5  # [B, C, 1, 1]

        # Equilibrium gain: push channels toward their equilibrium energy
        log_ratio = torch.log(self.equilibrium / energy + 1e-5)
        eq_gain = torch.exp(self.strength * torch.tanh(log_ratio))  # range [e^-s, e^s]

        # Combined modulation: attention + equilibrium adjustment
        out = x * attn * eq_gain
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = FEM(channels=128)
    output = model(input_tensor)
    print('=== FEM: Feature Equilibrium Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('equilibrium (first 5):', model.equilibrium[0, :5, 0, 0].detach().tolist())
    print('strength:', model.strength.item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
