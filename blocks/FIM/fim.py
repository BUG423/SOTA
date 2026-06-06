import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：FIM (Frequency Importance Module) —— 频率重要性模块

一、模块简介
卷积神经网络天然偏向于学习低频（平滑、全局）特征，而高频信息（边缘、
纹理、细节）往往在逐层传播中逐渐衰减。虽然这种低频偏好在一定程度上
提供了平移不变性和鲁棒性，但也导致了精细空间细节的丢失——这对需要
精确定位的任务（如分割、边缘检测）尤为不利。

FIM 的核心思想是：在频域（DCT域）中学习各频率分量的重要性权重，然后
据此对不同频率进行差异化增强。具体而言，FIM 首先通过二维离散余弦变换
（2D DCT）将特征分解为不同频率的分量；然后利用一个轻量级的频率重要性
网络学习每个频率分量对任务的贡献程度；最后通过频率重标定（重要频率增强、
冗余频率抑制）和逆变换回到空间域。与通道注意力（SENet）关注"哪些通道
重要"不同，FIM 关注的是"哪些频率分量重要"。

核心创新点：
1. DCT域特征调制：在频域而非通道/空间域进行特征重标定
2. 频率重要性学习：显式学习各频率分量的任务相关性
3. 频谱重标定：类似通道注意力的"频谱注意力"，但作用于频率维度
4. 高低频平衡：自适应调节高频细节和低频语义的贡献比例

二、结构设计
FIM 由以下子结构组成：
1. DCT 分解：
   - 对输入特征逐通道进行 2D DCT 变换
   - 将空间域特征转换为频谱表示
2. 频率重要性网络（Frequency Importance Network）：
   - 全局平均池化 → 1x1 压缩 → ReLU → 1x1 恢复 → Sigmoid
   - 输入频谱幅值，输出每个频率的重要性权重
3. 频率重标定：
   - 重要性权重 × 频谱
   - 对重要频率增强，对不重要频率抑制
4. IDCT 重建：
   - 2D 逆 DCT 变换回空间域
5. 空间精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 FIM（Frequency Importance Module）模块，通过在频域中学习
频率分量的重要性权重来实现特征增强。该模块利用离散余弦变换将特征分解
到频域，通过频率重要性网络评估各频率分量的任务相关性，并据此进行
差异化的频率重标定——增强判别性频率分量以提升特征表达质量，抑制冗余
频率分量以降低噪声。与通道或空间注意力互补，FIM 提供了第三维度的
调制机制。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要精细空间细节的任务（如分割、
超分辨率、边缘检测），以及对纹理信息敏感的场景（如纹理分类、缺陷检测）。
'''


class FIM(nn.Module):
    """FIM: Frequency Importance Module —— 频率重要性模块"""

    def __init__(self, channels: int, reduction: int = 4, freq_groups: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.freq_groups = freq_groups

        # Frequency importance network: learn importance per frequency group
        self.freq_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(freq_groups),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Learnable frequency group importance baseline
        self.freq_baseline = nn.Parameter(torch.ones(1, channels, freq_groups, freq_groups))

        # Spatial refinement after inverse DCT
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D Discrete Cosine Transform per channel.
        Uses torch.fft-based DCT: DCT = real(FFT of symmetrically extended signal)
        """
        B, C, H, W = x.shape

        # Create DCT basis using FFT
        # DCT-II: Reflect padding then FFT
        # Pad: reflect along both spatial dims
        x_pad = torch.cat([x, x.flip(2)], dim=2)                     # [B, C, 2H, W]
        x_pad = torch.cat([x_pad, x_pad.flip(3)], dim=3)             # [B, C, 2H, 2W]

        # FFT of real signal
        X = torch.fft.rfft2(x_pad.float())                           # [B, C, 2H, W+1]
        # Take real part (cosine components = DCT)
        dct_coeff = X.real
        # Truncate to original spatial size
        dct_coeff = dct_coeff[:, :, :H, :W]                          # [B, C, H, W]

        return dct_coeff

    def _idct2d(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate inverse 2D DCT via inverse FFT"""
        B, C, H, W = x.shape

        # Zero-pad to double size for inverse
        padded = F.pad(x, (0, W, 0, H), mode='constant', value=0)    # [B, C, 2H, 2W]

        # Inverse FFT
        X_complex = torch.complex(padded, torch.zeros_like(padded))
        x_rec = torch.fft.irfft2(X_complex)                          # [B, C, 2H, 2W]

        # Crop to original size
        x_rec = x_rec[:, :, :H, :W]                                  # [B, C, H, W]

        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Transform to frequency domain via 2D DCT
        freq = self._dct2d(x)                                        # [B, C, H, W]

        # 2. Downsample frequency to groups for importance learning
        freq_pooled = F.adaptive_avg_pool2d(freq, (self.freq_groups,
                                                    self.freq_groups))  # [B, C, G, G]

        # 3. Learn frequency importance weights
        importance = self.freq_importance(freq_pooled)                # [B, C, G, G]
        importance = importance * self.freq_baseline

        # 4. Upsample importance to full frequency resolution
        importance_full = F.interpolate(importance, size=(H, W),
                                         mode='bilinear', align_corners=False)  # [B, C, H, W]

        # 5. Frequency reweighting
        freq_reweighted = freq * importance_full                      # [B, C, H, W]

        # 6. Transform back to spatial domain
        spatial = self._idct2d(freq_reweighted)                      # [B, C, H, W]

        # 7. Spatial refinement
        spatial = self.spatial_refine(spatial)                       # [B, C, H, W]

        # 8. Refine and residual
        out = self.refine(spatial)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = FIM(channels=128)
    output = model(input_tensor)
    print('=== FIM: Frequency Importance Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
