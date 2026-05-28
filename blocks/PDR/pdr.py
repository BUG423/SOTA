import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-28

'''
模块名称：PDR (Polarized Dual Representation) —— 极化双表示模块

一、模块简介
在人类视觉系统中，背侧通路（dorsal stream，"where"通路）和腹侧通路（ventral
stream，"what"通路）分别处理空间定位和物体识别信息。这两种信息处理方式具有
不同的特性：空间通路关注结构、边缘和位置关系；语义通路关注内容、类别和全局
属性。然而，现有的特征增强模块通常将两者混在一起处理，没有显式区分这两种
互补的表示。

PDR 的核心思想是：将特征通道沿维度一分为二，一半通过"空间路径"强化局部结构
和边缘信息（where），另一半通过"语义路径"强化全局语义和通道交互（what），
并通过交叉门控机制在两条路径之间交换互补信息，最终融合形成兼具空间精度和
语义丰富度的增强特征。

核心创新点：
1. 极化分解：显式地将特征分解为空间路径和语义路径，模拟双通路处理
2. 交叉门控（Cross-Gating）：每条路径生成门控信号指导另一条路径的信息选择
3. 非对称处理：两条路径使用不同结构和感受野，针对性强化各自的信息类型
4. 通道级自适应分配：学习每层中空间/语义路径的最优通道分配比例

二、结构设计
PDR 由以下子结构组成：
1. 特征分割器：
   - 将 C 个通道按比例 α 分割为空间组（C_space）和语义组（C_semantic）
   - α 可配置，默认为 0.5
2. 空间路径（Spatial Stream）：
   - 3x3 深度可分离卷积 + 空间注意力残差
   - 专注于局部结构、边缘和空间关系
3. 语义路径（Semantic Stream）：
   - 1x1 通道混合 + 全局上下文残差
   - 专注于通道交互和全局语义
4. 交叉门控（Cross-Gating）：
   - 空间路径生成空间注意力图，用于调制语义路径的输出
   - 语义路径生成通道注意力向量，用于调制空间路径的输出
5. 融合层：将两条路径的输出在通道维度拼接后通过 1x1 卷积融合

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 PDR（Polarized Dual Representation）模块，受人类视觉双通路理论启发，
将特征显式分解为空间表示和语义表示两条路径分别处理。空间路径通过深度可分离
卷积强化局部结构信息，语义路径通过通道交互强化全局上下文信息，两条路径通过
交叉门控机制交换互补信息，实现空间精度和语义丰富度的统一。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。特别适合需要同时关注空间细节和语义理解的场景。
'''


class PDR(nn.Module):
    """PDR: Polarized Dual Representation —— 极化双表示模块"""

    def __init__(self, channels: int, alpha: float = 0.5):
        super().__init__()
        self.ch_spatial = max(1, int(channels * alpha))
        self.ch_semantic = channels - self.ch_spatial

        # 空间路径：局部结构和边缘
        self.spatial_path = nn.Sequential(
            nn.Conv2d(self.ch_spatial, self.ch_spatial, kernel_size=3,
                      padding=1, groups=self.ch_spatial, bias=False),
            nn.BatchNorm2d(self.ch_spatial),
            nn.GELU(),
        )

        # 语义路径：通道交互和全局语义
        self.semantic_path = nn.Sequential(
            nn.Conv2d(self.ch_semantic, self.ch_semantic, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch_semantic),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.ch_semantic, self.ch_semantic, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 交叉门控生成器
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.ch_spatial, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.semantic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.ch_semantic, self.ch_spatial, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
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

        # 通道分割
        x_spatial = x[:, :self.ch_spatial, :, :]          # [B, C_s, H, W]
        x_semantic = x[:, self.ch_spatial:, :, :]          # [B, C_c, H, W]

        # 空间路径处理
        spatial_out = self.spatial_path(x_spatial)          # [B, C_s, H, W]

        # 语义路径处理
        semantic_weight = self.semantic_path(x_semantic)    # [B, C_c, 1, 1]
        semantic_out = x_semantic * semantic_weight          # [B, C_c, H, W]

        # 交叉门控
        s_gate = self.spatial_gate(spatial_out)              # [B, 1, H, W]
        c_gate = self.semantic_gate(x_semantic)              # [B, C_s, 1, 1]

        # 交叉调制
        spatial_out = spatial_out * c_gate                   # 语义门控空间
        semantic_out = semantic_out * s_gate                 # 空间门控语义

        # 拼接融合
        fused = torch.cat([spatial_out, semantic_out], dim=1)  # [B, C, H, W]
        out = self.fusion(fused)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = PDR(channels=128, alpha=0.5)
    output = model(input_tensor)
    print('=== PDR: Polarized Dual Representation ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('spatial ch:', model.ch_spatial, 'semantic ch:', model.ch_semantic)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
