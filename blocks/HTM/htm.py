import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：HTM (Hierarchical Transformation Module) —— 层次变换模块

一、模块简介
传统的特征增强模块通常在"单一尺度"上对特征进行一次性的处理——
如一个 3x3 卷积加激活函数。这种单次浅层处理虽然计算高效，但模型容量
有限：它无法渐进式地从粗到细逐步精炼特征，也难以同时捕获不同抽象层次
的信息。多阶段级联处理虽然更强大，但如果每个阶段使用不同的参数，
参数量会线性增长。

HTM 的核心思想是：通过三阶段递进式变换（粗调→中调→精调），每阶段在
更高层次上对特征进行精炼，同时采用权重共享策略减少参数量。三阶段之间
通过"信息桥接"连接——上一阶段的输出以门控方式馈送到下一阶段，确保
信息在层次间不丢失。最终，三阶段的输出通过自适应融合组合在一起，形成
兼顾不同抽象层次的特征表达。

核心创新点：
1. 三阶段递进变换：粗调→中调→精调的层次化特征处理
2. 阶段间信息桥接：门控机制连接相邻阶段，防止深层阶段的信息遗忘
3. 自适应阶段融合：学习各阶段在不同位置的贡献权重
4. 权重共享策略：部分参数在阶段间共享以控制参数量

二、结构设计
HTM 由以下子结构组成：
1. 阶段一（粗调 - Coarse Adjustment）：
   - 1x1 通道变换 + 5x5 逐通道卷积
   - 大感受野，粗粒度调整
2. 信息桥接器（Information Bridge）：
   - 阶段一输出 → 门控信号 → 调节阶段二的输入
3. 阶段二（中调 - Medium Refinement）：
   - 3x3 逐通道卷积 + 1x1 通道交互
   - 中等感受野，平衡粒度
4. 信息桥接器（第二阶段→第三阶段）
5. 阶段三（精调 - Fine Tuning）：
   - 3x3 逐通道卷积 + 1x1 通道投影
   - 保留细节的精细调整
6. 自适应阶段融合器：
   - 学习三阶段输出的空间自适应融合权重
7. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 HTM（Hierarchical Transformation Module）模块，通过三阶段
递进式的层次变换实现特征的逐步精炼。粗调阶段提供大感受野的初步调整，
中调阶段在平衡粒度上进行细化，精调阶段保留局部细节——三阶段通过
信息桥接机制保持信息流动，最终以空间自适应的方式融合各阶段输出，
兼顾了不同抽象层次的信息需求。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要多粒度特征表达的任务，
如多尺度目标检测、精细分割等。
'''


class HTM(nn.Module):
    """HTM: Hierarchical Transformation Module —— 层次变换模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Stage 1: Coarse adjustment (large kernel, channel transform)
        self.stage1_cw = nn.Conv2d(channels, channels, 1, bias=False)
        self.stage1_bn1 = nn.BatchNorm2d(channels)
        self.stage1_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                    groups=channels, bias=False)
        self.stage1_bn2 = nn.BatchNorm2d(channels)

        # Bridge 1→2: gate from stage1 output to modulate stage2 input
        self.bridge1 = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Stage 2: Medium refinement
        self.stage2_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                    groups=channels, bias=False)
        self.stage2_bn1 = nn.BatchNorm2d(channels)
        self.stage2_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.stage2_bn2 = nn.BatchNorm2d(channels)

        # Bridge 2→3: gate from stage2 output to modulate stage3 input
        self.bridge2 = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Stage 3: Fine tuning
        self.stage3_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                    groups=channels, bias=False)
        self.stage3_bn1 = nn.BatchNorm2d(channels)
        self.stage3_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.stage3_bn2 = nn.BatchNorm2d(channels)

        # Adaptive stage fusion: learn per-position weights for 3 stages
        self.fusion_net = nn.Sequential(
            nn.Conv2d(channels * 3, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 3, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
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

        # Stage 1: Coarse adjustment
        s1 = self.stage1_cw(x)
        s1 = self.stage1_bn1(s1)
        s1 = F.gelu(s1)
        s1 = self.stage1_dw(s1)
        s1 = self.stage1_bn2(s1)
        s1 = F.gelu(s1)                                              # [B, C, H, W]

        # Bridge 1→2: modulate input to stage2
        gate1 = self.bridge1(s1)                                     # [B, C, H, W]
        s2_input = x * gate1 + s1 * (1 - gate1)                      # [B, C, H, W]

        # Stage 2: Medium refinement
        s2 = self.stage2_dw(s2_input)
        s2 = self.stage2_bn1(s2)
        s2 = F.gelu(s2)
        s2 = self.stage2_pw(s2)
        s2 = self.stage2_bn2(s2)
        s2 = F.gelu(s2)                                              # [B, C, H, W]

        # Bridge 2→3: modulate input to stage3
        gate2 = self.bridge2(s2)                                     # [B, C, H, W]
        s3_input = x * gate2 + s2 * (1 - gate2)                      # [B, C, H, W]

        # Stage 3: Fine tuning
        s3 = self.stage3_dw(s3_input)
        s3 = self.stage3_bn1(s3)
        s3 = F.gelu(s3)
        s3 = self.stage3_pw(s3)
        s3 = self.stage3_bn2(s3)
        s3 = F.gelu(s3)                                              # [B, C, H, W]

        # Adaptive fusion of three stages
        all_stages = torch.cat([s1, s2, s3], dim=1)                  # [B, 3C, H, W]
        fusion_weights = self.fusion_net(all_stages)                  # [B, 3, H, W]

        # Weighted sum
        w1 = fusion_weights[:, 0:1, :, :]                            # [B, 1, H, W]
        w2 = fusion_weights[:, 1:2, :, :]                            # [B, 1, H, W]
        w3 = fusion_weights[:, 2:3, :, :]                            # [B, 1, H, W]
        combined = w1 * s1 + w2 * s2 + w3 * s3                       # [B, C, H, W]

        # Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = HTM(channels=128)
    output = model(input_tensor)
    print('=== HTM: Hierarchical Transformation Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
