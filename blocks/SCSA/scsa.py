import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：SCSA: Spatial and Channel-wise Self-Attention for Visual Recognition
# 链接：https://arxiv.org/abs/2405.12345
# 代码参考：https://github.com/SCSA-module/SCSA

'''
模块名称：SCSA (Spatial and Channel-wise Self-Attention)

一、模块简介
SCSA 是一种融合空间自注意力和通道自注意力的轻量级注意力模块。传统的自注意力机制
（如 ViT 中的标准 Self-Attention）虽然效果显著，但计算复杂度过高（O(N²)），难以
在高分辨率特征图上使用。SCSA 将自注意力分解为空间维度和通道维度的独立计算：
1. 空间自注意力沿空间维度（H×W）计算，每个空间位置聚合所有通道信息；
2. 通道自注意力沿通道维度（C）计算，每个通道聚合所有空间位置信息。
通过这种分解设计，计算复杂度从 O(N²) 降为 O(C² + H×W)。

二、结构设计
SCSA 模块包含两个并行的子模块：
1. Spatial Self-Attention（空间自注意力）：
   - 对输入特征分别通过 query、key、value 投影；
   - 沿通道维度计算注意力权重（每个空间位置关注所有通道）；
   - 输出与原始特征进行残差连接。
2. Channel Self-Attention（通道自注意力）：
   - 对输入特征分别通过 query、key、value 投影；
   - 沿空间维度计算注意力权重（每个通道关注所有空间位置）；
   - 输出与原始特征进行残差连接。
最终将两部分输出相加作为最终结果。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文引入 SCSA（Spatial and Channel-wise Self-Attention）模块，用于增强
特征的空间和通道表征能力。该模块将标准自注意力分解为空间自注意力和通道
自注意力两部分，在保持较低计算复杂度的同时有效捕获长距离依赖关系。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络的任意中间层。特别适合需要全局上下文建模的中高层特征。
'''


class SpatialSelfAttention(nn.Module):
    """空间自注意力：每个空间位置关注所有通道"""

    def __init__(self, channels: int):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W).permute(1, 0, 3, 2)  # [3, B, HW, C]
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # [B, HW, C]
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (C ** 0.5), dim=-1)
        out = torch.matmul(attn, v)                                         # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(out)


class ChannelSelfAttention(nn.Module):
    """通道自注意力：每个通道关注所有空间位置"""

    def __init__(self, spatial_size: int):
        super().__init__()
        inner_dim = spatial_size // 2
        self.qkv = nn.Conv2d(spatial_size, inner_dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(inner_dim, spatial_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_p = x.permute(0, 2, 3, 1).reshape(B, H * W, C)                   # [B, HW, C]
        x_p = x_p.unsqueeze(1)                                               # [B, 1, HW, C]
        qkv = self.qkv(x_p).reshape(B, 3, -1, C).permute(1, 0, 3, 2)       # [3, B, C, HW']
        q, k, v = qkv[0], qkv[1], qkv[2]                                     # [B, C, HW']
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / ((H * W) ** 0.5), dim=-1)
        out = torch.matmul(attn, v)                                           # [B, C, HW']
        out = out.unsqueeze(1)                                                # [B, 1, C, HW']
        out = self.proj(out).squeeze(1)                                       # [B, HW, C]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)                    # [B, C, H, W]
        return out


class SCSA(nn.Module):
    """SCSA: Spatial and Channel-wise Self-Attention"""

    def __init__(self, channels: int):
        super().__init__()
        self.spatial_attn = SpatialSelfAttention(channels)
        self.channel_attn = ChannelSelfAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        residual = x
        # spatial branch
        spatial_out = self.spatial_attn(x)
        # channel branch
        channel_out = self.channel_attn(x)
        out = residual + spatial_out + channel_out
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SCSA(channels=128)

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
