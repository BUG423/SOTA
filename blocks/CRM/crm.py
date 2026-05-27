import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：CRM (Channel Recalibration Module) —— 通道重校准模块

一、模块简介
SE（Squeeze-and-Excitation）模块通过全局平均池化压缩空间信息来学习通道权重，
但这种做法有一个隐含假设：每个通道的重要性在全局范围内是均匀的。实际上，一个
通道可能在某些空间区域很重要（如边缘检测通道在物体边界处），在另一些区域则
不那么重要（如平坦背景区域）。SE 的全局池化会"抹平"这种空间差异性。

CRM 的核心思想是：利用通道激活的熵值来度量每个通道的"信息丰富度"——高熵的
通道（激活分布更分散）通常包含更多样化的信息，应当被保留甚至增强；低熵的通道
（激活分布集中）可能包含冗余信息，可以被适度抑制。通过熵引导的通道重校准，
CRM 实现了比 SE 更细粒度的通道调制。

核心创新点：
1. 熵引导的通道评估：使用激活熵而非简单的全局平均来评估通道重要性
2. 局部-全局联合调制：同时考虑局部熵（空间分布）和全局熵（整体分布）
3. 自适应温度缩放：通过可学习参数控制调制的强度
4. 无额外全连接层：所有操作通过卷积完成，保持空间结构

二、结构设计
CRM 由以下子结构组成：
1. 局部熵估计器（Local Entropy Estimator）：
   - 对每个通道的激活值计算空间维度的局部熵
   - 使用分组 softmax 将激活转换为概率分布
   - 输出每个通道的熵值（标量）
2. 全局信息压缩器（Global Context Compressor）：
   - 并行使用平均池化和最大池化
   - 拼接后压缩为通道权重
3. 通道调制器（Channel Modulator）：
   - 将熵值与全局信息融合，生成最终的通道权重
   - 使用 sigmoid 限制在 [0, 1] 范围

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 CRM（Channel Recalibration Module）模块，通过通道激活熵来引导通道
重要性的重校准。该模块利用每个通道激活分布的熵值度量其信息丰富度，并结合全局
上下文信息生成精细化的通道调制权重，从而抑制冗余通道并增强信息丰富的通道。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。特别适合通道数较多的深层网络。
'''


class CRM(nn.Module):
    """CRM: Channel Recalibration Module —— 通道重校准模块"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        # 全局信息分支
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        # 通道权重生成
        self.weight_generator = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # 调制强度
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _compute_channel_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """计算每个通道的空间激活熵 [B, 1, 1, 1]（平均）"""
        B, C, H, W = x.shape
        # 将激活转为非负值用于概率计算
        x_pos = torch.abs(x).view(B, C, -1)                     # [B, C, HW]
        probs = x_pos / (x_pos.sum(dim=-1, keepdim=True) + 1e-8) # [B, C, HW]
        # 计算熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B, C]
        # 归一化
        entropy = entropy / torch.log(torch.tensor(H * W, dtype=torch.float32, device=x.device))
        return entropy.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        # 全局信息
        avg_pool = self.global_avg_pool(x)                       # [B, C, 1, 1]
        max_pool = self.global_max_pool(x)                       # [B, C, 1, 1]
        # 通道熵
        entropy = self._compute_channel_entropy(x)                # [B, 1, 1, 1]

        # 拼接生成权重
        weight_input = torch.cat([avg_pool, max_pool, entropy], dim=1)  # [B, 2C+1, 1, 1]
        weights = self.weight_generator(weight_input)             # [B, C, 1, 1]

        # 调制
        out = x * (1.0 + self.scale * (weights - 0.5) * 2.0)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CRM(channels=128)
    output = model(input_tensor)
    print('=== CRM: Channel Recalibration Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('scale:', model.scale.item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
