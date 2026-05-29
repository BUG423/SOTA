import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-29

'''
模块名称：PGM (Progressive Gating Module) —— 渐进式门控模块

一、模块简介
在深度神经网络中，门控机制（如 SE、CBAM）广泛用于自适应地调制特征响应。
然而，现有的门控模块通常采用"单次决策"的方式——通过固定的计算流程一次性
生成门控权重，缺乏对特征的逐步精炼能力。单次门控难以同时兼顾全局语义和
局部细节，可能导致信息调制不够精准。

PGM 的核心思想是：将特征调制分解为三个渐进式的门控阶段，从粗到细逐步精炼。
第一阶段采用全局池化生成粗粒度的通道门控（类似 SE），完成初步的通道筛选；
第二阶段利用跨通道空间统计（均值图 + 最大值图）生成中等粒度的混合门控，
在空间维度上提供更精细的调制；第三阶段通过空间卷积作用于前两阶段的输出，
生成最细粒度的空间感知门控。三个阶段级联处理，前一阶段的输出作为后一阶段
的输入，并通过可学习的通道级权重将各阶段输出自适应融合。

核心创新点：
1. 渐进式级联门控：从全局→混合→局部，逐步提高门控粒度，而非一次性决策
2. 异构门控设计：三个阶段使用不同粒度和结构的信息源，互补增强
3. 通道级自适应融合：每个通道独立学习三个阶段的重要性权重
4. 级联残差增量：每个阶段在前一阶段基础上进行增量式精炼

二、结构设计
PGM 由以下子结构组成：
1. 粗粒度通道门控（Stage 1）：
   - 全局平均池化 → 降维 → GELU → 升维 → Sigmoid
   - 生成 [B, C, 1, 1] 的通道注意力权重
   - 完成粗粒度的通道重要性评估
2. 中粒度统计门控（Stage 2）：
   - 沿通道维度计算均值图和最大值图（各 1 个通道）
   - 拼接为 [B, 2, H, W]，通过两层 1x1 卷积生成 [B, C, H, W] 门控图
   - 在保留空间结构的前提下进行中等粒度调制
3. 细粒度空间门控（Stage 3）：
   - 3x3 卷积 → 降维 → GELU → 3x3 卷积 → Sigmoid
   - 作用于前两阶段的输出，进行最细粒度的空间感知调制
4. 自适应阶段融合：
   - 每组通道独立学习三个阶段权重 [3, C, 1, 1]
   - Softmax 归一化后加权组合三个阶段输出
   - 最终通过残差连接与原始输入相加

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 PGM（Progressive Gating Module）模块，通过渐进式多阶段门控机制
实现从粗到细的特征自适应精炼。该模块包含三个异构门控阶段：全局通道门控
进行粗粒度通道选择，混合统计门控进行中粒度空间-通道调制，空间感知门控
进行细粒度局部调制。三个阶段级联处理，并通过通道级可学习权重自适应融合
各阶段输出，使特征增强过程兼具全局视野和局部精度。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要多层次特征调制的场景，如
细粒度识别、小目标检测等。
'''


class PGM(nn.Module):
    """PGM: Progressive Gating Module —— 渐进式门控模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Stage 1: Global channel gate (coarse)
        self.gate1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Stage 2: Mixed spatial-statistics gate (medium)
        self.gate2 = nn.Sequential(
            nn.Conv2d(2, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Stage 3: Spatial-aware gate (fine)
        self.gate3 = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # Channel-wise learnable stage fusion weights
        self.stage_w = nn.Parameter(torch.ones(3, channels, 1, 1) / 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # Stage 1: Coarse global channel gate
        g1 = self.gate1(x)                                    # [B, C, 1, 1]
        s1 = x * g1

        # Stage 2: Medium gate using avg+max spatial statistics
        avg_map = x.mean(dim=1, keepdim=True)                  # [B, 1, H, W]
        max_map, _ = x.max(dim=1, keepdim=True)                # [B, 1, H, W]
        stats2 = torch.cat([avg_map, max_map], dim=1)          # [B, 2, H, W]
        g2 = self.gate2(stats2)                                 # [B, C, H, W]
        s2 = s1 * g2

        # Stage 3: Fine spatial-channel gate on refined features
        g3 = self.gate3(s2)                                     # [B, C, H, W]
        s3 = s2 * g3

        # Adaptive channel-wise fusion of all stages
        w = F.softmax(self.stage_w, dim=0)                     # [3, C, 1, 1]
        out = x + w[0] * s1 + w[1] * s2 + w[2] * s3
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = PGM(channels=128)
    output = model(input_tensor)
    print('=== PGM: Progressive Gating Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    w = F.softmax(model.stage_w, dim=0)
    print('stage_weights:', w[:, 0, 0, 0].detach().tolist())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
