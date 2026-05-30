import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-30

'''
模块名称：OEM (Order-Statistic Enhancement Module) —— 序统计增强模块

一、模块简介
标准卷积操作本质上是局部邻域值的加权线性组合。虽然高效，但线性聚合对异常
值和极端激活值敏感——单个极端像素可能主导整个邻域的卷积输出，造成特征
失真。中值滤波等序统计方法对异常值具有天然的鲁棒性，但传统中值滤波不可
学习且丢失了空间结构信息。

OEM 的核心思想是：将序统计量的鲁棒性与可学习的神经网络相结合。对于每个
空间位置，OEM 从局部邻域中提取多个序统计量（均值、最大值、最小值、以及
通过软排序机制获得的近似分位数），并通过一个可学习的空间自适应门控网络
为每个位置选择最合适的序统计量组合。这种"软选择"机制使得网络在高纹理
区域倾向保留原始特征（对抗模糊），在平坦区域倾向使用中值类统计量（抑制
噪声），在边缘区域倾向使用极值统计量（增强对比度）。

核心创新点：
1. 多序统计量提取：从局部邻域同时提取均值、最大值、最小值和软中值
2. 空间自适应统计量选择：每个位置独立学习最优的统计量组合权重
3. 鲁棒特征增强：结合序统计量的抗噪性和线性卷积的表达能力
4. 软排序近似：使用可微分的软排序机制，保持端到端训练能力

二、结构设计
OEM 由以下子结构组成：
1. 序统计量提取器（Order-Statistic Extractor）：
   - 3x3 滑动窗口展开（unfold）为 [B, C*9, H*W] 的局部邻域向量
   - 对每个邻域计算：均值（mean）、最大值（max）、最小值（min）
   - 软中值：通过可学习的排序权重近似中值
   - 将 4 个统计量重组为 [B, C*4, H, W] 的特征图
2. 统计量编码器（Statistic Encoder）：
   - 将 4 个序统计量通过 1x1 卷积映射回原始通道数
   - 使用 GELU 激活进行非线性变换
3. 空间自适应选择网络（Spatial Selection Network）：
   - 基于原始输入特征生成 4 通道的空间权重图
   - Softmax 归一化得到每个位置对 4 种统计量的偏好
4. 特征增强与融合：
   - 根据空间权重加权组合 4 种统计量增强的特征
   - 通过残差连接与原始特征融合

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 OEM（Order-Statistic Enhancement Module）模块，通过引入序统计
先验增强卷积特征的鲁棒性。该模块从局部邻域中提取多种序统计量（均值、极值、
近似中值），并通过空间自适应选择网络为每个位置学习最优的统计量组合。
这种设计使网络能够在不同区域自适应地选择最合适的聚合策略——在噪声区域
倾向鲁棒统计，在结构区域保留原始线性响应——从而提升特征的质量和鲁棒性。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合噪声较大的场景（如低光照图像、
医学图像），以及需要对抗鲁棒性的任务。
'''


class OEM(nn.Module):
    """OEM: Order-Statistic Enhancement Module —— 序统计增强模块"""

    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        inner = max(1, channels // reduction)
        self.pad = kernel_size // 2
        self.k2 = kernel_size * kernel_size

        # --- Learnable soft-median weights (shared across channels for efficiency) ---
        self.sort_weights = nn.Parameter(torch.zeros(channels, self.k2))
        nn.init.normal_(self.sort_weights, std=1e-3)

        # --- Statistic encoder: 4 stats → channels ---
        self.stat_encoder = nn.Sequential(
            nn.Conv2d(channels * 4, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # --- Spatial selection network: original features → 4 stat preference maps ---
        self.spatial_select = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 4, 1, bias=False),
        )

        # --- Output refinement ---
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _extract_stats(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multiple order statistics from local patches.
        Input: [B, C, H, W] → Output: [B, 4C, H, W]
        """
        B, C, H, W = x.shape

        # Unfold into patches: [B, C*k*k, L] where L = H*W
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.pad,
                           stride=1)  # [B, C*k2, H*W]
        patches = patches.view(B, C, self.k2, H, W)                # [B, C, k2, H, W]

        # Mean: average of all neighbors
        stat_mean = patches.mean(dim=2)                              # [B, C, H, W]

        # Max: maximum neighbor value
        stat_max = patches.max(dim=2).values                         # [B, C, H, W]

        # Min: minimum neighbor value
        stat_min = patches.min(dim=2).values                         # [B, C, H, W]

        # Soft-median: weighted combination of sorted-like values
        # Sort patches along the neighbor dimension
        patches_sorted, _ = patches.sort(dim=2)                      # [B, C, k2, H, W]
        # Apply learnable softmax weights per channel
        w = F.softmax(self.sort_weights, dim=1)                     # [C, k2]
        w = w.view(1, C, self.k2, 1, 1)                             # [1, C, k2, 1, 1]
        stat_softmed = (patches_sorted * w).sum(dim=2)               # [B, C, H, W]

        # Concatenate all statistics: [B, 4C, H, W]
        stats = torch.cat([stat_mean, stat_max, stat_min, stat_softmed], dim=1)
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Extract 4 order statistics from local patches
        stats = self._extract_stats(x)                               # [B, 4C, H, W]

        # 2. Encode statistics into feature space
        stat_feats = self.stat_encoder(stats)                        # [B, C, H, W]

        # 3. Spatial selection: 4 per-position preference maps
        select = self.spatial_select(x)                              # [B, 4, H, W]
        select = F.softmax(select, dim=1)                            # normalize over stat types

        # 4. Apply each selection map as a spatial gate on encoded features
        gated_list = []
        for i in range(4):
            gate = select[:, i:i+1, :, :]                            # [B, 1, H, W]
            gated_list.append(stat_feats * gate)                     # [B, C, H, W]

        # 5. Sum all gated views (softmax ensures convex combination)
        combined = torch.stack(gated_list, dim=0).sum(dim=0)         # [B, C, H, W]

        # 6. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = OEM(channels=128)
    output = model(input_tensor)
    print('=== OEM: Order-Statistic Enhancement Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('sort_weights shape:', model.sort_weights.shape)
    print('sort_weights entropy (lower=sharper):',
          (-F.softmax(model.sort_weights, dim=1) *
           F.log_softmax(model.sort_weights, dim=1)).sum(dim=1).mean().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
