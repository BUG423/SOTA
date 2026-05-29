import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-29

'''
模块名称：IGM (Information Gathering Module) —— 信息汇聚模块

一、模块简介
标准卷积操作受限于固定的局部感受野，每个位置的输出仅依赖于其邻域内的输入。
虽然堆叠多层可以逐步扩大感受野，但浅层特征难以获取足够的上下文信息。
Non-local 和自注意力机制虽然能捕获长程依赖，但计算开销大，不适合作为
即插即用模块在网络中大量使用。

IGM 的核心思想是：在保持轻量计算的前提下，通过多尺度信息汇聚机制为每个
空间位置收集不同范围邻域的上下文信息。具体而言，IGM 使用两个不同核大小的
深度可分离卷积分支（3x3 和 5x5），分别从局部邻域和扩展邻域汇聚信息，
并通过一个空间自适应权重网络学习每个位置对两个尺度信息的依赖程度，
实现"按需汇聚"——在高纹理区域倾向局部信息，在平坦区域倾向更广的上下文。

核心创新点：
1. 多尺度信息汇聚：双分支深度可分离卷积同时收集局部和扩展邻域信息
2. 空间自适应汇聚权重：每个空间位置独立学习两个尺度的混合比例
3. 深度可分离设计：通过逐通道空间卷积 + 逐点通道混合，保持极低参数量
4. 软汇聚策略：通过 softmax 归一化的空间权重实现平滑的尺度过渡

二、结构设计
IGM 由以下子结构组成：
1. 局部汇聚分支（Local Gather）：
   - 3x3 逐通道卷积 + BatchNorm + 1x1 逐点卷积
   - 从 3x3 邻域汇聚局部细节信息
2. 扩展汇聚分支（Extended Gather）：
   - 5x5 逐通道卷积 + BatchNorm + 1x1 逐点卷积
   - 从 5x5 邻域汇聚更广泛的上下文信息
3. 空间权重网络（Spatial Weight Network）：
   - 1x1 卷积将 C 通道映射为 2 通道（对应两个尺度）
   - Softmax 归一化得到每个位置的双尺度混合权重
4. 汇聚融合与精炼：
   - 加权组合两个分支的汇聚结果
   - 通过 1x1 卷积 + BatchNorm 精炼融合特征
   - 残差连接输出

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 IGM（Information Gathering Module）模块，通过多尺度信息汇聚
机制以极低的计算开销为特征图注入多范围上下文信息。该模块采用双分支深度
可分离卷积分别从不同大小的邻域汇聚信息，并通过空间自适应权重网络学习
每个位置的尺度偏好，实现局部细节与扩展上下文的动态平衡。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要多尺度上下文但计算预算有限的
场景，如轻量化模型、实时检测等。
'''


class IGM(nn.Module):
    """IGM: Information Gathering Module —— 信息汇聚模块"""

    def __init__(self, channels: int):
        super().__init__()

        # Local gather branch: 3x3 depthwise separable
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.local_bn = nn.BatchNorm2d(channels)
        self.local_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Extended gather branch: 5x5 depthwise separable
        self.ext_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                groups=channels, bias=False)
        self.ext_bn = nn.BatchNorm2d(channels)
        self.ext_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Spatial weight network: learn per-position scale preferences
        self.spatial_weight = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, 2, 1, bias=False),
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
        # Local gathering: 3x3 neighborhood
        g_local = self.local_dw(x)
        g_local = self.local_bn(g_local)
        g_local = F.gelu(g_local)
        g_local = self.local_pw(g_local)                        # [B, C, H, W]

        # Extended gathering: 5x5 neighborhood
        g_ext = self.ext_dw(x)
        g_ext = self.ext_bn(g_ext)
        g_ext = F.gelu(g_ext)
        g_ext = self.ext_pw(g_ext)                              # [B, C, H, W]

        # Spatial-adaptive scale weights
        w = self.spatial_weight(x)                               # [B, 2, H, W]
        w = F.softmax(w, dim=1)                                  # normalize over scales
        w_local = w[:, 0:1, :, :]                                # [B, 1, H, W]
        w_ext = w[:, 1:2, :, :]                                  # [B, 1, H, W]

        # Weighted gathering
        gathered = w_local * g_local + w_ext * g_ext            # [B, C, H, W]

        # Refine and residual
        out = self.refine(gathered)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = IGM(channels=128)
    output = model(input_tensor)
    print('=== IGM: Information Gathering Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
