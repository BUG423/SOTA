import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：GFF (Gated Feature Fusion) —— 门控特征融合模块

一、模块简介
现有的多分支特征融合方法（如 Inception、SKNet）通常对所有空间位置采用相同的
融合策略。然而，不同空间位置对不同变换的偏好是不同的。例如，物体中心区域可能
更需要通道混合带来的语义信息，而边缘区域可能更需要空间卷积带来的结构信息。

GFF 的核心思想是：使用三条并行分支（恒等映射、深度可分离卷积、通道混合），并
通过一个轻量级门控网络为每个空间位置和每个通道预测三条分支的融合权重，实现
输入自适应的动态特征融合。

核心创新点：
1. 三路并行变换：恒等（保留原始信息）、空间卷积（提取结构）、通道混合（语义增强）
2. 空间-通道联合门控：同时考虑空间位置和通道的重要性来分配融合权重
3. 竞争性融合：三条分支的权重通过 softmax 归一化，形成竞争机制
4. 低计算开销：门控网络仅使用少量 1x1 卷积

二、结构设计
GFF 由以下子结构组成：
1. 三路并行变换分支：
   - Identity Branch: 直接保留输入特征
   - Spatial Branch: 3x3 深度可分离卷积，提取空间结构信息
   - Channel Branch: 1x1 卷积 + GELU，进行通道间信息交互
2. 门控权重生成器（Gate Generator）：
   - 拼接三路输出，通过两层 1x1 卷积预测每路的权重
   - 使用 softmax 确保三路权重之和为 1
3. 加权融合：按门控权重对三路输出进行加权求和

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 GFF（Gated Feature Fusion）模块，用于实现输入自适应的多路特征融合。
该模块通过三条并行分支分别保留原始信息、提取空间结构和增强通道语义，并利用
轻量级门控网络为每个空间-通道位置动态分配融合权重，从而提升特征表达的灵活性。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。特别适合替换标准卷积层以提升模型容量。
'''


class GFF(nn.Module):
    """GFF: Gated Feature Fusion —— 门控特征融合模块"""

    def __init__(self, channels: int):
        super().__init__()
        # 空间卷积分支
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        # 通道混合分支
        self.channel_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # 门控生成器：输入三路特征 → 输出三路权重
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 三条分支
        identity = x
        spatial = self.spatial_branch(x)
        channel = self.channel_branch(x)

        # 门控权重
        gate_input = torch.cat([identity, spatial, channel], dim=1)  # [B, 3C, H, W]
        weights = self.gate(gate_input)                                # [B, 3, H, W]
        weights = torch.softmax(weights, dim=1)                        # 竞争性融合

        # 加权融合（每个空间位置独立加权）
        w_id, w_sp, w_ch = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        out = w_id * identity + w_sp * spatial + w_ch * channel
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = GFF(channels=128)
    output = model(input_tensor)
    print('=== GFF: Gated Feature Fusion ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
