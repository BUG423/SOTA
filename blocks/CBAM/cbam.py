import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：CBAM: Convolutional Block Attention Module
# 链接：https://arxiv.org/abs/1807.06521 (ECCV 2018)
# 代码参考：https://github.com/luuuyi/CBAM.PyTorch

'''
模块名称：CBAM (Convolutional Block Attention Module)

一、模块简介
CBAM 是 Woo 等人在 ECCV 2018 上提出的轻量级注意力模块。该模块从通道和空间两个维度
计算注意力权重，通过串联的方式增强卷积神经网络的特征表达能力。CBAM 计算开销很小，
可以无缝嵌入到任何 CNN 架构中，显著提升分类和检测性能。

二、结构设计
CBAM 包含两个串行的子模块：
1. Channel Attention (通道注意力)：对输入特征图分别做全局最大池化和全局平均池化，
   然后通过共享的 MLP 生成通道注意力权重。
2. Spatial Attention (空间注意力)：对通道注意力加权后的特征沿通道维度分别做
   最大池化和平均池化，拼接后通过卷积层生成空间注意力权重。
最终将两部分权重依次与原始特征相乘，得到增强后的特征。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文在骨干网络中引入 CBAM（Convolutional Block Attention Module）注意力模块。
该模块通过串行的通道注意力和空间注意力机制，自适应地增强关键特征通道和重要空间
位置的表征，从而提升模型对判别性特征的关注能力。"

四、适用任务
适用于图像分类、目标检测、实例分割等视觉任务，可作为即插即用模块嵌入
ResNet、MobileNet 等 CNN 主干网络的任意卷积层之后。
'''


class ChannelAttention(nn.Module):
    """通道注意力子模块"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力子模块"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CBAM(channels=128, reduction=16, spatial_kernel=7)

    output = model(input_tensor)

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
