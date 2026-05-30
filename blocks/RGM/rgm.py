import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-30

'''
模块名称：RGM (Reciprocal Guidance Module) —— 互惠引导模块

一、模块简介
在卷积神经网络中，特征提取通常依赖单一的计算路径，缺乏来自不同视角的互补
信息交互。虽然多分支结构（如 Inception、ResNeXt）通过并行分支增加了表示
多样性，但各分支之间通常是独立计算、最后简单相加，缺少在中间阶段的相互
引导和修正机制。

RGM 的核心思想是：构建"通道-空间"双分支互惠引导结构。通道分支通过全局
池化捕获通道间依赖关系，空间分支通过深度可分离卷积建模局部空间结构。
两个分支在计算过程中相互引导——通道分支的输出作为空间分支的调制信号，
告诉空间分支"哪些通道应该被关注"；空间分支的输出作为通道分支的调制
信号，告诉通道分支"哪些位置更重要"。这种双向信息交互使得两个分支能够
互相补充、互相纠正，最终产生更丰富、更平衡的特征表示。

核心创新点：
1. 互惠引导机制：双分支之间进行双向信息调制，而非简单的并行+相加
2. 异构分支设计：通道分支和空间分支采用不同粒度的操作，互补强化
3. 自适应引导强度：通过可学习参数控制互惠引导的力度
4. 迭代精炼可选：支持多次互惠引导迭代（可配置迭代次数）

二、结构设计
RGM 由以下子结构组成：
1. 通道感知分支（Channel Branch）：
   - 全局自适应平均池化 → 两层 1x1 卷积（降维+升维） → Sigmoid
   - 生成 [B, C, 1, 1] 的通道注意力权重
   - 关注"哪些通道重要"
2. 空间感知分支（Spatial Branch）：
   - 3x3 逐通道卷积（depthwise） + 1x1 逐点卷积（pointwise）
   - 生成 [B, C, H, W] 的空间调制图
   - 关注"哪些位置重要"
3. 互惠引导模块（Reciprocal Guidance）：
   - 通道注意力显著化（squeeze → sigmoid）后作用于空间分支的输入
   - 空间调制图池化为通道描述子后作用于通道分支
   - 双向信息流动，互相增强
4. 融合输出：
   - 两个分支的增强输出通过可学习权重自适应融合
   - 残差连接保留原始特征

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 RGM（Reciprocal Guidance Module）模块，通过通道-空间双分支
互惠引导机制增强特征表示。该模块包含一个通道感知分支和一个空间感知分支，
两个分支在计算过程中互相提供调制信号：通道分支为空间分支提供通道重要性
先验，空间分支为通道分支提供空间显著性反馈。这种双向引导机制使得两个
异构建模视角能够相互补充，共同产生更丰富的特征表示。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要同时关注语义信息（通道）和
空间细节的任务，如细粒度分类、小目标检测等。
'''


class RGM(nn.Module):
    """RGM: Reciprocal Guidance Module —— 互惠引导模块"""

    def __init__(self, channels: int, reduction: int = 8, num_iterations: int = 1):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_iterations = num_iterations

        # Channel branch: global pooling → MLP → channel attention
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial branch: depthwise + pointwise conv
        self.spatial_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)
        self.spatial_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Spatial-to-channel guidance: compress spatial map → channel descriptor
        self.s2c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel-to-spatial guidance: channel weights → spatial modulation
        self.c2s = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Learnable branch fusion weights (per-channel)
        self.fusion_w = nn.Parameter(torch.ones(2, channels, 1, 1) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        feat = x

        for _ in range(self.num_iterations):
            # Channel branch: which channels matter?
            ch_attn = self.channel_mlp(feat)                          # [B, C, 1, 1]
            ch_out = feat * ch_attn                                    # [B, C, H, W]

            # Spatial branch: which positions matter?
            sp_feat = self.spatial_dw(feat)
            sp_feat = self.spatial_bn(sp_feat)
            sp_feat = F.gelu(sp_feat)
            sp_feat = self.spatial_pw(sp_feat)                        # [B, C, H, W]

            # Reciprocal guidance: channel → spatial
            c2s_guide = self.c2s(ch_out)                               # [B, C, H, W]
            sp_out = sp_feat * c2s_guide                               # [B, C, H, W]

            # Reciprocal guidance: spatial → channel
            s2c_guide = self.s2c(sp_out)                               # [B, C, 1, 1]
            ch_out_refined = ch_out * s2c_guide                        # [B, C, H, W]

            # Fuse both branches for next iteration
            w = F.softmax(self.fusion_w, dim=0)                        # [2, C, 1, 1]
            feat = w[0] * ch_out_refined + w[1] * sp_out               # [B, C, H, W]

        # Final output with residual
        return feat + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = RGM(channels=128, num_iterations=2)
    output = model(input_tensor)
    print('=== RGM: Reciprocal Guidance Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    w = F.softmax(model.fusion_w, dim=0)
    print('fusion_weights (ch+branch avg):', w.mean(dim=(1, 2, 3)).detach().tolist())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
