import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-03

'''
模块名称：AGM (Adaptive Granularity Module) —— 自适应粒度模块

一、模块简介
在视觉特征处理中，"粒度"（granularity）指特征分析的空间分辨率粗细
程度。细粒度处理（如标准 3x3 卷积）保留更多空间细节但感受野较小，
适合纹理丰富区域；粗粒度处理（如下采样后的大核卷积）提供更大的感受野
和语义上下文但丢失空间分辨率，适合平坦或语义一致区域。传统多尺度方法
（如 FPN、ASPP）在不同粒度上分别处理然后拼接或相加，计算开销大且各
粒度之间缺少对"何处该用何种粒度"的显式建模。

AGM 的核心思想是：学习一个空间自适应的"粒度偏好图"，根据每个位置的
特征内容决定其最优处理粒度——在纹理丰富的细节区域使用细粒度处理以保留
精度，在语义一致的区域使用粗粒度处理以获取更广的上下文并节省计算。
与固定多分支结构不同，AGM 的粒度切换是连续的而非离散的——每个位置的
输出是粗细两个粒度分支结果的软插值，插值系数由粒度偏好图决定。

核心创新点：
1. 空间自适应粒度选择：逐位置学习最优的处理粒度，粗细连续可调
2. 粒度偏好图：通过轻量网络从输入特征中推断每个位置的粒度偏好
3. 粗细双分支互补：粗粒度（下采样+大核+上采样）提供语义上下文，
   细粒度（原始分辨率）保留空间细节
4. 分辨率自适应池化：粗粒度分支采用自适应池化到固定尺寸，确保对不同
   输入分辨率鲁棒

二、结构设计
AGM 由以下子结构组成：
1. 粒度偏好网络（Granularity Preference Network）：
   - 1x1 卷积压缩 → BN → GELU → 1x1 卷积 → Sigmoid
   - 输出 [B, 1, H, W] 的粒度偏好图
   - 接近 1 → 偏好粗粒度，接近 0 → 偏好细粒度
2. 粗粒度分支（Coarse Branch）：
   - 自适应平均池化降采样到固定尺寸
   - 5x5 逐通道卷积 + 1x1 逐点卷积
   - 双线性插值上采样恢复原始分辨率
3. 细粒度分支（Fine Branch）：
   - 3x3 逐通道卷积 + 1x1 逐点卷积
   - 保持原始分辨率
4. 粒度引导融合：
   - 粗粒度输出 * 粒度偏好 + 细粒度输出 * (1 - 粒度偏好)
5. 精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 AGM（Adaptive Granularity Module）模块，通过空间自适应的
粒度选择机制实现内容感知的特征处理。该模块从输入特征中学习一个粒度
偏好图，指示每个空间位置适合的处理粒度——纹理区域偏好细粒度以保留
细节，平坦区域偏好粗粒度以获取语义上下文。粗细两个分支的连续插值
确保粒度过渡自然平滑，避免硬切换带来的块效应。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要同时处理多尺度目标或包含
异质纹理区域的场景，如高分辨率遥感图像、街景分析等。
'''


class AGM(nn.Module):
    """AGM: Adaptive Granularity Module —— 自适应粒度模块"""

    def __init__(self, channels: int, reduction: int = 4, coarse_size: int = 16):
        super().__init__()
        inner = max(1, channels // reduction)
        self.coarse_size = coarse_size

        # Granularity preference network
        self.granularity_net = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # Coarse branch: downsample → large kernel → upsample
        self.coarse_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                    groups=channels, bias=False)
        self.coarse_bn = nn.BatchNorm2d(channels)
        self.coarse_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Fine branch: original resolution
        self.fine_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.fine_bn = nn.BatchNorm2d(channels)
        self.fine_pw = nn.Conv2d(channels, channels, 1, bias=False)

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

        # 1. Granularity preference map: per-position coarse vs fine preference
        granularity = self.granularity_net(x)                        # [B, 1, H, W]

        # 2. Coarse branch: downsample → process → upsample
        coarse_h = max(2, self.coarse_size)
        coarse_w = max(2, self.coarse_size)
        x_down = F.adaptive_avg_pool2d(x, (coarse_h, coarse_w))
        coarse = self.coarse_dw(x_down)
        coarse = self.coarse_bn(coarse)
        coarse = F.gelu(coarse)
        coarse = self.coarse_pw(coarse)                              # [B, C, cs, cs]
        coarse = F.interpolate(coarse, size=(H, W), mode='bilinear',
                               align_corners=False)                  # [B, C, H, W]

        # 3. Fine branch: process at original resolution
        fine = self.fine_dw(x)
        fine = self.fine_bn(fine)
        fine = F.gelu(fine)
        fine = self.fine_pw(fine)                                    # [B, C, H, W]

        # 4. Granularity-guided fusion
        # granularity → 1: prefer coarse; → 0: prefer fine
        combined = granularity * coarse + (1 - granularity) * fine   # [B, C, H, W]

        # 5. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = AGM(channels=128)
    output = model(input_tensor)
    print('=== AGM: Adaptive Granularity Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    granularity = model.granularity_net(input_tensor)
    print('granularity mean:', granularity.mean().item(),
          'std:', granularity.std().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
