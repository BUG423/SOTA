import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Efficient Multi-Scale Attention Module with Cross-Spatial Learning
# 链接：https://arxiv.org/abs/2305.13563 (ICASSP 2023)
# 代码参考：https://github.com/Alexanderzlwang/EMA

'''
模块名称：EMA (Efficient Multi-Scale Attention)

一、模块简介
EMA 是 Ouyang 等人在 ICASSP 2023 上提出的高效多尺度注意力模块。该模块旨在解决
传统注意力机制（如 SE、CBAM）仅关注单一尺度特征的问题，通过对通道进行分组并执行
跨空间维度的多尺度交互，在几乎不增加计算量的情况下捕获更丰富的上下文信息。

EMA 的核心创新在于：
1. 将通道分成多个子组，每组独立学习空间注意力；
2. 通过跨空间交互（cross-spatial learning）在水平/垂直方向编码位置信息；
3. 使用 2D 全局平均池化补充全局上下文。

二、结构设计
EMA 模块包含以下子结构：
1. 通道分组（Group Division）：将输入特征沿通道维度分为 G 组，每组独立处理。
2. 并行分支：
   - 1x1 卷积分支：提取局部特征并进行通道间的跨组交互；
   - 3x3 卷积分支：捕获多尺度空间特征。
3. 跨空间注意力聚合：
   - 对 1x1 分支的输出沿水平/垂直方向做自适应平均池化；
   - 将编码的方向信息与 3x3 分支输出融合；
   - 通过 Sigmoid 生成空间注意力图。
4. 跨通道交互：将两组分支的注意力图加权融合后，通过全局平均池化引入
   通道维度的信息补充。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文引入 EMA（Efficient Multi-Scale Attention）注意力模块。该模块通过通道分组
与跨空间学习机制，在多个尺度上捕获空间注意力信息，并以极小的计算开销增强特征
表达的质量。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。由于计算量极小，特别适合轻量化模型或实时检测场景。
'''


class EMA(nn.Module):
    """EMA: Efficient Multi-Scale Attention Module"""

    def __init__(self, channels: int, factor: int = 32):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0, "channels must be divisible by factor"
        self.softmax = nn.Softmax(dim=-1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1,
                                 stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3,
                                 stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        group_x = x.reshape(B * self.groups, -1, H, W)           # [B*G, C//G, H, W]
        x_h = self.avg_pool(group_x).view(B * self.groups, -1)   # [B*G, C//G]
        x_w = group_x.mean(dim=-2)                                # [B*G, C//G, W]

        # 1x1 conv branch
        x_1x1 = self.conv1x1(group_x)                             # [B*G, C//G, H, W]
        x_1x1_h = x_1x1.mean(dim=-1)                              # [B*G, C//G, H]
        x_1x1_w = x_1x1.mean(dim=-2)                              # [B*G, C//G, W]

        x_h_s = self.softmax(x_1x1_h)                             # [B*G, C//G, H]
        x_w_s = self.softmax(x_1x1_w)                             # [B*G, C//G, W]

        # 3x3 conv branch
        x_3x3 = self.conv3x3(group_x)                             # [B*G, C//G, H, W]
        x_3x3 = self.gn(x_3x3)

        x_11 = torch.einsum('bch,bcw->bhw', x_h_s, x_w).view(B * self.groups, 1, H, W)
        x_12 = torch.einsum('bcw,bch->bwh', x_w_s, x_h).view(B * self.groups, 1, W, H)
        x_12 = x_12.permute(0, 1, 3, 2)                           # [B*G, 1, H, W]

        spatial_weight = torch.sigmoid(x_11 + x_12)
        group_x = group_x * spatial_weight + x_3x3 * (1 - spatial_weight)
        group_x = group_x.view(B, -1, H, W)

        return group_x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = EMA(channels=128, factor=32)

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
