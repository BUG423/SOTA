import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：QEM (Quantile Enhancement Module) —— 分位数增强模块

一、模块简介
在深度神经网络的特征图中，激活值的分布常常呈现长尾或偏态特征——
少数通道/位置的激活值极高（可能是真正的强特征信号），而大量通道/位置
的激活值集中在较低水平。传统的批归一化（Batch Normalization）使用均值
和标准差进行归一化，对长尾分布和离群值敏感——极端值会显著影响均值和
方差的估计，导致归一化后的特征分布偏离理想状态。

QEM 的核心思想是：使用分位数（Quantile）统计量替代矩统计量（均值和
方差）进行特征归一化和增强。具体而言，QEM 估计每个通道的特征分布的分位数
（中位数、四分位数、十分位数等），基于分位数进行鲁棒的归一化（对离群值
不敏感），然后利用分位数之间的差异（如四分位距 IQR）评估特征的"信息
丰富度"——分布越分散（分位距大），信息越丰富，应被增强；分布越集中
（分位距小），信息越贫乏，应被抑制。

核心创新点：
1. 分位数归一化：用中位数和四分位距替代均值和标准差，对离群值鲁棒
2. 分位数驱动的增强：利用分位距作为信息丰富度的度量来驱动增强强度
3. 多分位数统计：超过中位数和IQR的粗粒度，多个分位数提供细粒度分布刻画
4. 自适应软阈值：基于分位数的软阈值函数实现自适应特征筛选

二、结构设计
QEM 由以下子结构组成：
1. 分位数估计器（Quantile Estimator）：
   - 通过排序或可微分近似估计多个分位数（如 Q10, Q25, Q50, Q75, Q90）
   - 使用软排序的连续松弛保证可微分性
2. 鲁棒归一化：
   - 中位数中心化 + IQR 缩放
   - (x - median) / (IQR + eps)
3. 分布特征提取：
   - 分位距（Q75-Q25, Q90-Q10）作为分布分散度指标
   - 偏度（(Q75-Q50) vs (Q50-Q25)）作为分布对称性指标
4. 增强系数生成：
   - 从分布特征通过 MLP 生成每通道的增强/抑制系数
5. 空间精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 QEM（Quantile Enhancement Module）模块，通过分位数统计量
替代传统矩统计量进行鲁棒的特征归一化和增强。该模块基于分位数（而非均值
和方差）进行归一化以抵抗离群值干扰，并利用分位距作为通道信息丰富度
的度量来指导自适应增强——信息丰富（分布分散）的通道被增强，信息贫乏
（分布集中）的通道被抑制，实现分布感知的特征重标定。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合激活分布偏态严重、存在极端离群值
的深层网络层，以及对分布偏移需要鲁棒处理的场景（如域自适应、迁移学习）。
'''


class QEM(nn.Module):
    """QEM: Quantile Enhancement Module —— 分位数增强模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_quantiles: int = 5):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_quantiles = num_quantiles
        # Quantile levels: e.g., [0.1, 0.25, 0.5, 0.75, 0.9] for 5 quantiles
        self.register_buffer(
            'q_levels',
            torch.linspace(0.1, 0.9, num_quantiles)
        )

        # Distribution feature extractor: quantile stats → channel modulation
        # Features: num_quantiles quantile values + IQR + skewness
        self.dist_analyzer = nn.Sequential(
            nn.Conv2d(channels * (num_quantiles + 2), inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Learnable per-channel quantile normalization parameters
        self.norm_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.norm_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Spatial processing
        self.spatial_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)
        self.spatial_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _estimate_quantiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-channel quantiles via differentiable soft sorting.
        Returns quantile values for each channel.
        """
        B, C, H, W = x.shape
        N = H * W

        # Sort per channel (this is differentiable via sort)
        x_flat = x.view(B, C, N)                                     # [B, C, N]
        x_sorted, _ = torch.sort(x_flat, dim=2)                      # [B, C, N]

        # Interpolate at quantile positions
        quantile_values = []
        for q in self.q_levels:
            idx = q * (N - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, N - 1)
            frac = idx - idx_low

            # Linear interpolation for differentiable quantile
            q_val = (x_sorted[:, :, idx_low] * (1 - frac) +
                     x_sorted[:, :, idx_high] * frac)                # [B, C]
            quantile_values.append(q_val)

        # Stack: [B, num_quantiles, C]
        quantiles = torch.stack(quantile_values, dim=1)               # [B, num_q, C]

        return quantiles

    def _get_distribution_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-channel distribution statistics from quantiles"""
        B, C, H, W = x.shape

        quantiles = self._estimate_quantiles(x)                       # [B, num_q, C]

        # Extract key statistics
        # q25 = quantiles[:, 1, :] (index 1 for 0.25), q50 = quantiles[:, 2, :], q75 = quantiles[:, 3, :]
        idx = {0.1: 0, 0.3: 0, 0.5: self.num_quantiles // 2, 0.7: self.num_quantiles - 2, 0.9: self.num_quantiles - 1}
        q10 = quantiles[:, 0, :]                                      # [B, C] ~ Q10
        q25 = quantiles[:, 1, :] if self.num_quantiles >= 3 else q10  # [B, C] ~ Q25
        q50 = quantiles[:, self.num_quantiles // 2, :]                # [B, C] ~ Q50 (median)
        q75 = quantiles[:, -2, :] if self.num_quantiles >= 4 else quantiles[:, -1, :]  # [B, C] ~ Q75
        q90 = quantiles[:, -1, :]                                     # [B, C] ~ Q90

        # IQR (Inter-Quartile Range): measure of spread
        iqr = q75 - q25 + 1e-8                                       # [B, C]

        # Skewness proxy: asymmetry of upper vs lower tail
        skew = ((q90 - q50) - (q50 - q10)) / (iqr + 1e-8)            # [B, C]

        # Expand to spatial dimensions
        iqr_sp = iqr.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, C, H, W]
        skew_sp = skew.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Also expand quantile values for spatial context
        quantile_sp = []
        for q_idx in range(self.num_quantiles):
            q_sp = quantiles[:, q_idx, :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            quantile_sp.append(q_sp)

        return torch.cat(quantile_sp + [iqr_sp, skew_sp], dim=1)     # [B, C*(num_q+2), H, W]

    def _robust_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using median and IQR (robust to outliers)"""
        B, C, H, W = x.shape

        quantiles = self._estimate_quantiles(x)                       # [B, num_q, C]
        median = quantiles[:, self.num_quantiles // 2, :]             # [B, C] Q50

        # IQR
        q25 = quantiles[:, 1, :] if self.num_quantiles >= 3 else quantiles[:, 0, :]
        q75 = quantiles[:, -2, :] if self.num_quantiles >= 4 else quantiles[:, -1, :]
        iqr = q75 - q25 + 1e-8                                       # [B, C]

        # Normalize: (x - median) / IQR
        median = median.view(B, C, 1, 1)
        iqr = iqr.view(B, C, 1, 1)
        normalized = (x - median) / iqr                               # [B, C, H, W]

        return normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Robust quantile normalization
        normalized = self._robust_normalize(x)                       # [B, C, H, W]
        normalized = normalized * self.norm_scale + self.norm_bias

        # 2. Extract distribution statistics for enhancement guidance
        dist_stats = self._get_distribution_stats(x)                  # [B, C*(num_q+2), H, W]

        # 3. Generate enhancement coefficients from distribution
        enhance_coeff = self.dist_analyzer(dist_stats)                # [B, C, H, W]

        # 4. Apply enhancement
        enhanced = normalized * enhance_coeff                         # [B, C, H, W]

        # 5. Spatial processing
        spatial = self.spatial_dw(enhanced)
        spatial = self.spatial_bn(spatial)
        spatial = F.gelu(spatial)
        spatial = self.spatial_pw(spatial)                           # [B, C, H, W]

        # 6. Refine and residual
        out = self.refine(spatial)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = QEM(channels=128)
    output = model(input_tensor)
    print('=== QEM: Quantile Enhancement Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    quantiles = model._estimate_quantiles(input_tensor)
    print('quantiles shape:', quantiles.shape)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
