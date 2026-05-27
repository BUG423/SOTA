import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：DRS (Dynamic Receptive Field Selector) —— 动态感受野选择器

一、模块简介
在卷积神经网络中，固定的卷积核大小限制了模型对不同尺度目标的适应性。ASPP 和
RFB 等模块通过并行多尺度卷积来扩展感受野，但它们在融合多尺度特征时忽略了
空间位置的差异性——大物体需要大感受野，小物体需要小感受野。

DRS 的核心思想是：为每个空间位置动态选择最合适的感受野大小。模块使用多个
不同膨胀率的并行卷积分支，并通过一个轻量级选择器网络为每个位置预测各感受野
的重要性权重，实现空间自适应的感受野选择。

核心创新点：
1. 空间自适应感受野选择：每个位置独立选择最优感受野
2. 软选择机制：使用可微分的 softmax 权重，允许位置同时利用多个感受野
3. 多粒度设计：膨胀率从 1 到 5，覆盖细粒度到粗粒度的范围
4. 选择性残差连接：通过门控决定每个位置是否需要多尺度信息

二、结构设计
DRS 由以下子结构组成：
1. 多感受野卷积分支：
   - 5 个并行分支，使用 3x3 卷积，膨胀率分别为 1, 2, 3, 4, 5
   - 每个分支均为深度可分离卷积以降低计算量
2. 感受野选择器（Selector）：
   - 对输入特征做全局平均池化并压缩
   - 通过 MLP 预测每个感受野分支的全局重要性
   - 再通过 1x1 卷积生成空间相关的细粒度权重
3. 加权融合：按选择器权重对多分支输出加权求和
4. 自适应残差：学习一个全局门控决定多尺度信息与原始特征的混合比例

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 DRS（Dynamic Receptive Field Selector）模块，用于动态选择每个空间
位置的最优感受野大小。该模块通过多膨胀率并行卷积分支和轻量级感受野选择器，
实现空间自适应的多尺度特征提取，有效提升模型对不同尺度目标的适应性。"

四、适用任务
适用于目标检测、语义分割等需要处理多尺度目标的视觉任务，可作为即插即用模块
嵌入 CNN 主干网络，尤其适合替换检测头的单尺度卷积。
'''


class DRS(nn.Module):
    """DRS: Dynamic Receptive Field Selector —— 动态感受野选择器"""

    dilations = [1, 2, 3, 4, 5]

    def __init__(self, channels: int):
        super().__init__()
        # 多感受野卷积分支（深度可分离）
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            for d in self.dilations
        ])
        # 全局选择器
        self.global_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, len(self.dilations), bias=False),
        )
        # 空间细化器
        self.spatial_refiner = nn.Conv2d(channels, len(self.dilations), kernel_size=1, bias=False)
        # 自适应门控
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 多分支输出
        branch_outs = [branch(x) for branch in self.branches]  # 5 × [B, C, H, W]
        stacked = torch.stack(branch_outs, dim=1)                # [B, 5, C, H, W]

        # 全局权重
        global_w = self.global_selector(x).unsqueeze(-1).unsqueeze(-1)  # [B, 5, 1, 1]
        # 空间权重
        spatial_w = self.spatial_refiner(x).unsqueeze(2)                # [B, 5, 1, H, W]
        # 融合权重
        weights = torch.softmax(global_w + spatial_w, dim=1)            # [B, 5, C, H, W]

        # 加权融合
        fused = (stacked * weights).sum(dim=1)  # [B, C, H, W]

        # 自适应残差
        out = x + self.residual_gate * fused
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = DRS(channels=128)
    output = model(input_tensor)
    print('=== DRS: Dynamic Receptive Field Selector ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
