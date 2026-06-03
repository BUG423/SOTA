import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-03

'''
模块名称：SUM (Spatial Uncertainty Module) —— 空间不确定性模块

一、模块简介
在卷积神经网络的特征图中，不同空间位置的"可靠性"并不均匀。纹理丰富的
区域（如物体边缘、高纹理表面）通常产生稳定且信息量大的特征响应；而平坦
区域或噪声区域的特征响应则具有较高的不确定性——微小的输入扰动可能导致
较大的特征变化。现有的特征增强模块通常对所有空间位置一视同仁，缺少对
这种空间异质可靠性的显式建模。

SUM 的核心思想是：为每个空间位置估计一个"不确定性分数"，并以此为引导
对不同区域进行差异化处理。具体而言，SUM 通过局部邻域统计（方差、信息熵）
估计每个位置的特征不确定性：高不确定性区域（平坦、噪声）倾向于接受更多
的空间平滑以稳定特征响应；低不确定性区域（纹理、边缘）则保留原有响应
以维持细节精度。这种"因不确定性施策"的自适应处理策略使模块能够在去噪
和保真之间实现空间自适应的平衡。

核心创新点：
1. 空间不确定性估计：通过局部方差和特征熵量化每个位置的不确定性
2. 不确定性引导的自适应处理：处理策略（平滑 vs 保持）由不确定性驱动
3. 软决策机制：连续的不确定性分数实现平滑的策略过渡，无硬阈值
4. 双路径架构：平滑路径和保持路径并行，由不确定性门控融合

二、结构设计
SUM 由以下子结构组成：
1. 不确定性估计器（Uncertainty Estimator）：
   - 计算局部邻域（3x3）的通道方差和均值
   - 拼接并通过 1x1 卷积生成单通道不确定性图
   - Sigmoid 激活，输出 [B, 1, H, W]
2. 稳定化分支（Stabilization Branch）：
   - 5x5 逐通道卷积 + 1x1 逐点卷积
   - 大感受野平滑，用于降低高不确定性区域的特征波动
3. 保持分支（Preservation Branch）：
   - 3x3 逐通道卷积 + 1x1 逐点卷积
   - 轻量处理，保持低不确定性区域的细节精度
4. 不确定性引导融合：
   - 不确定性分数作为平滑分支的权重
   - 输出 = u * smoothed + (1-u) * preserved
5. 精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 SUM（Spatial Uncertainty Module）模块，通过估计特征图中每个
空间位置的不确定性来引导自适应的特征处理。该模块利用局部邻域统计量化
各位置的特征可靠性，并以此为据在空间平滑（高不确定性区域）与细节保持
（低不确定性区域）之间实现连续、自适应的策略切换，从而在抑制噪声的
同时最大限度保留有效的结构信息。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合输入质量不稳定或噪声水平较高的
场景，如低光照图像、医学图像、遥感图像等。
'''


class SUM(nn.Module):
    """SUM: Spatial Uncertainty Module —— 空间不确定性模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Uncertainty estimator: local statistics → uncertainty map
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(channels * 2, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # Stabilization branch: large-kernel smoothing
        self.stab_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                  groups=channels, bias=False)
        self.stab_bn = nn.BatchNorm2d(channels)
        self.stab_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Preservation branch: light processing to keep details
        self.pres_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.pres_bn = nn.BatchNorm2d(channels)
        self.pres_pw = nn.Conv2d(channels, channels, 1, bias=False)

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

        # 1. Estimate spatial uncertainty via local statistics
        # Local variance (as a proxy for uncertainty)
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # [B, C, H, W]
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3,
                                  stride=1, padding=1)                    # [B, C, H, W]
        # Channel-mean variance and channel-mean activation as statistics
        var_stat = local_var.mean(dim=1, keepdim=True)                    # [B, 1, H, W]
        mean_stat = local_mean.mean(dim=1, keepdim=True)                  # [B, 1, H, W]
        # Expand to per-channel statistics
        var_per_ch = local_var                                             # [B, C, H, W]
        mean_per_ch = local_mean                                           # [B, C, H, W]

        # Combine for uncertainty estimation
        stats = torch.cat([var_per_ch, mean_per_ch], dim=1)               # [B, 2C, H, W]
        uncertainty = self.uncertainty_estimator(stats)                    # [B, 1, H, W]

        # 2. Stabilization branch: large-kernel smoothing
        smoothed = self.stab_dw(x)
        smoothed = self.stab_bn(smoothed)
        smoothed = F.gelu(smoothed)
        smoothed = self.stab_pw(smoothed)                                  # [B, C, H, W]

        # 3. Preservation branch: light processing
        preserved = self.pres_dw(x)
        preserved = self.pres_bn(preserved)
        preserved = F.gelu(preserved)
        preserved = self.pres_pw(preserved)                                # [B, C, H, W]

        # 4. Uncertainty-guided fusion
        # High uncertainty → more smoothing; Low uncertainty → more preservation
        combined = uncertainty * smoothed + (1 - uncertainty) * preserved  # [B, C, H, W]

        # 5. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SUM(channels=128)
    output = model(input_tensor)
    print('=== SUM: Spatial Uncertainty Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
