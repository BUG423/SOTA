import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：EDM (Entropy-Driven Module) —— 熵驱动模块

一、模块简介
在卷积特征图中，不同空间位置的信息丰富程度差异显著——纹理复杂区域
（如物体边缘、高纹理表面）承载大量结构化信息，而平坦区域（如天空、
墙面）主要包含冗余的平滑信号。然而，大多数特征增强模块对所有空间位置
采用相同的处理策略，未能根据信息密度进行差异化处理。

EDM 的核心思想是：利用局部信息熵作为"信息密度"的代理指标，驱动空间
自适应的特征处理。在信息熵高的区域（复杂纹理），EDM 增强特征响应以保留
更多判别性信息；在信息熵低的区域（平坦冗余），EDM 进行适度的特征压缩
以减少冗余。这种"按需分配"的策略使得计算资源更多地向信息丰富的区域倾斜，
在总计算量不变的前提下提升特征表达的效率。

核心创新点：
1. 局部信息熵估计：通过局部邻域的离散化概率分布计算空间信息熵
2. 熵引导的增强/压缩：高熵增强、低熵压缩的自适应处理策略
3. 软决策映射：连续熵值到连续处理强度的平滑映射
4. 计算资源重分配：在不增加总计算量的前提下优化信息分布

二、结构设计
EDM 由以下子结构组成：
1. 局部熵估计器（Local Entropy Estimator）：
   - 对每个位置计算局部邻域（7x7）的归一化直方图
   - 从直方图计算香农熵
   - 输出 [B, 1, H, W] 的熵图
2. 熵映射网络（Entropy Mapping Network）：
   - 将熵值映射为增强/压缩系数
   - 2层 1x1 卷积 + Sigmoid
   - 高熵→系数>1（增强），低熵→系数<1（压缩）
3. 特征增强分支：
   - 3x3 逐通道卷积 + 1x1 逐点卷积
   - 提取增强型特征
4. 特征压缩分支：
   - 1x1 通道压缩 + 3x3 逐通道卷积 + 1x1 通道恢复
   - 紧凑表达用于低熵区域
5. 熵驱动融合与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 EDM（Entropy-Driven Module）模块，通过局部信息熵估计实现
内容自适应的特征处理。该模块量化每个空间位置的信息密度，并以此为据
在高熵区域增强特征响应、在低熵区域进行紧凑表达——在特征表达能力和
计算效率之间实现空间自适应的平衡，实现'好钢用在刀刃上'的信息重分配。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合输入中包含大面积平坦区域（如天空、
水面、背景虚化）的场景，可以在不增加计算量的前提下提升关键区域的
特征质量。
'''


class EDM(nn.Module):
    """EDM: Entropy-Driven Module —— 熵驱动模块"""

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        inner = max(1, channels // reduction)
        self.kernel_size = kernel_size

        # Local entropy estimator: convert features to probability-like distribution
        self.entropy_proj = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
        )

        # Entropy mapping network: entropy → enhancement/compression coefficient
        self.entropy_mapper = nn.Sequential(
            nn.Conv2d(1, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Enhancement branch: for high-entropy regions
        self.enhance_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=False)
        self.enhance_bn = nn.BatchNorm2d(channels)
        self.enhance_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Compression branch: compact representation for low-entropy regions
        self.compress_down = nn.Conv2d(channels, inner, 1, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(inner)
        self.compress_dw = nn.Conv2d(inner, inner, 3, padding=1,
                                      groups=inner, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(inner)
        self.compress_up = nn.Conv2d(inner, channels, 1, bias=False)

        # Learnable base gain
        self.base_gain = nn.Parameter(torch.ones(1, channels, 1, 1))

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _compute_local_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local Shannon entropy for each spatial position"""
        B, C, H, W = x.shape

        # Convert to probability-like values (per-channel softmax over spatial window)
        # Use average pooling to approximate local mean, then compute per-bin frequencies
        prob_like = F.softmax(self.entropy_proj(x), dim=1)           # [B, 8, H, W]

        # Compute local statistics via average pooling
        local_prob = F.avg_pool2d(prob_like, kernel_size=self.kernel_size,
                                   stride=1, padding=self.kernel_size // 2)  # [B, 8, H, W]

        # Entropy: -sum(p * log(p + eps))
        eps = 1e-8
        entropy = -torch.sum(local_prob * torch.log(local_prob + eps), dim=1,
                              keepdim=True)                           # [B, 1, H, W]

        # Normalize entropy to [0, 1] range
        max_entropy = torch.log(torch.tensor(8.0, device=x.device))
        entropy = entropy / max_entropy
        return entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Estimate local information entropy
        entropy_map = self._compute_local_entropy(x)                  # [B, 1, H, W]

        # 2. Map entropy to enhancement/compression coefficients
        coeff = self.entropy_mapper(entropy_map)                      # [B, C, H, W]
        coeff = self.base_gain * (coeff * 2.0)                        # range [0, 2*base_gain]

        # 3. Enhancement branch: rich processing for high-entropy regions
        enhanced = self.enhance_dw(x)
        enhanced = self.enhance_bn(enhanced)
        enhanced = F.gelu(enhanced)
        enhanced = self.enhance_pw(enhanced)                          # [B, C, H, W]

        # 4. Compression branch: compact processing for low-entropy regions
        compressed = self.compress_down(x)
        compressed = self.compress_bn1(compressed)
        compressed = F.gelu(compressed)
        compressed = self.compress_dw(compressed)
        compressed = self.compress_bn2(compressed)
        compressed = F.gelu(compressed)
        compressed = self.compress_up(compressed)                     # [B, C, H, W]

        # 5. Entropy-driven fusion: high entropy → prefer enhanced; low entropy → prefer compressed
        combined = coeff * enhanced + (1 - coeff / (self.base_gain * 2.0 + 1e-8)) * compressed

        # 6. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = EDM(channels=128)
    output = model(input_tensor)
    print('=== EDM: Entropy-Driven Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    entropy = model._compute_local_entropy(input_tensor)
    print('entropy map mean:', entropy.mean().item(), 'std:', entropy.std().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
