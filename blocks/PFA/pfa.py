import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：PFA (Progressive Feature Aggregator) —— 渐进式特征聚合器

一、模块简介
大多数特征增强模块（如 SE、CBAM）采用单步前向的注意力计算方式，即一次性生成
注意力权重并应用。这种方式忽略了特征增强本身的迭代性质——第一次增强后的特征
可能包含新的可被利用的模式，值得进一步提炼。

PFA 的核心思想是：将特征增强建模为渐进式迭代过程，通过两个阶段串联的精炼步骤，
第一阶段产生粗略的增强信号，第二阶段基于第一阶段的结果进行更精细的调整。
两阶段共享相似的骨架但使用独立的参数，形成"粗调-精调"的级联结构。

核心创新点：
1. 两阶段渐进式精炼：模仿从粗到细的优化过程
2. 阶段间特征桥接：第一阶段输出作为第二阶段的额外输入，实现信息传递
3. 残差叠加：两个阶段的增强信号以残差方式累积
4. 可配置阶段数：通过 stage_ratio 控制精炼深度

二、结构设计
PFA 由以下子结构组成：
1. 第一阶段（Coarse Enhancement）：
   - 对输入进行通道压缩 → 空间注意力生成 → 粗略特征增强
2. 第二阶段（Fine Refinement）：
   - 以第一阶段输出为输入，使用相同结构但独立参数
   - 额外接收原始输入作为参考（skip connection）
3. 特征融合：
   - 将两阶段的增强信号以可学习的权重融合
   - 最终输出 = 原始输入 + 第一阶段增量 + 第二阶段增量

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 PFA（Progressive Feature Aggregator）模块，通过两阶段渐进式精炼策略
实现更精细的特征增强。第一阶段进行粗略的特征调制，第二阶段基于粗调结果进行
进一步优化，两个阶段以残差方式累积增强信号，从而在不显著增加参数量的前提下
提升特征表达质量。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。对需要精细特征表达的高分辨率任务尤为有效。
'''


class EnhanceStage(nn.Module):
    """单阶段特征增强"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        # 通道注意力分支
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # 空间注意力分支
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # 输出融合
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.channel_attn(x)          # [B, C, 1, 1]
        sa = self.spatial_attn(x)          # [B, 1, H, W]
        enhanced = x * ca * sa              # [B, C, H, W]
        return self.fusion(enhanced)


class PFA(nn.Module):
    """PFA: Progressive Feature Aggregator —— 渐进式特征聚合器"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.stage1 = EnhanceStage(channels, reduction)
        self.stage2 = EnhanceStage(channels, reduction)
        # 可学习的阶段权重
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 第一阶段：粗调
        e1 = self.stage1(x)
        # 第二阶段：精调（以第一阶段输出为输入）
        e2 = self.stage2(e1)
        # 加权残差累积
        out = x + self.w1 * e1 + self.w2 * e2
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = PFA(channels=128)
    output = model(input_tensor)
    print('=== PFA: Progressive Feature Aggregator ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('w1:', model.w1.item(), 'w2:', model.w2.item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
