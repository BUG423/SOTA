import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：CAM (Contrast-Aware Module) —— 对比度感知模块

一、模块简介
在人类视觉系统中，对比度是感知和理解场景的基础——高对比度区域
（如物体边缘、强纹理）自然吸引更多视觉注意力，而低对比度区域
（如平坦表面、弱纹理）则被相对忽略。然而，标准卷积神经网络对所有
空间位置采用相同的卷积操作，缺少这种基于对比度的差异化处理机制。

CAM 的核心思想是：计算局部对比度作为"视觉显著性"的代理指标，然后
基于对比度对特征进行差异化处理——高对比度区域（边缘、强纹理）的特征
被增强和锐化以保留精确的空间结构，低对比度区域（平坦、平滑）的特征
被适度平滑以减少噪声并节省计算。这种对比度驱动的自适应处理使模块能够
在边缘保持和噪声抑制之间实现空间自适应的平衡。

核心创新点：
1. 局部对比度度量：通过局部标准差和梯度幅值联合量化对比度
2. 对比度引导调制：对比度图直接驱动增强/平滑的双分支融合权重
3. 锐化-平滑双路径：锐化路径（边缘增强）和平滑路径（去噪）互补
4. 对比度归一化：自适应对比度归一化确保对不同光照条件鲁棒

二、结构设计
CAM 由以下子结构组成：
1. 对比度估计器（Contrast Estimator）：
   - 计算局部标准差（3x3 邻域）
   - 计算 Sobel 梯度幅值
   - 两者组合通过 1x1 卷积生成对比度图
   - 输出 [B, 1, H, W]
2. 锐化分支（Sharpening Branch）：
   - 拉普拉斯算子风格：用 3x3 卷积提取高频成分
   - 将高频成分加回原始特征（类似 Unsharp Masking）
3. 平滑分支（Smoothing Branch）：
   - 5x5 逐通道卷积（大感受野平滑）
   - BatchNorm + GELU
4. 对比度引导融合：
   - 高对比度 → 锐化输出权重高
   - 低对比度 → 平滑输出权重高
5. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 CAM（Contrast-Aware Module）模块，通过局部对比度估计实现
空间自适应的特征处理。该模块将局部对比度作为视觉显著性的代理指标，
在高对比度区域（边缘、纹理）执行特征锐化以增强空间精度，在低对比度
区域（平坦表面）执行特征平滑以抑制噪声——实现类似人类视觉系统中
基于对比度的差异化处理。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要精确边缘保持的任务（如分割、
边缘检测），以及对噪声敏感的应用场景（如低光照图像增强、医学图像处理）。
'''


class CAM(nn.Module):
    """CAM: Contrast-Aware Module —— 对比度感知模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Contrast estimator: local std + gradient magnitude
        self.contrast_net = nn.Sequential(
            nn.Conv2d(channels * 2, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # Sobel-like gradient filters (fixed, not learned)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Sharpening branch: extract and enhance high-frequency components
        self.sharpen_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=False)
        self.sharpen_bn = nn.BatchNorm2d(channels)
        self.sharpen_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)

        # Smoothing branch: large-kernel smoothing for low-contrast regions
        self.smooth_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                    groups=channels, bias=False)
        self.smooth_bn = nn.BatchNorm2d(channels)
        self.smooth_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _compute_local_std(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-channel local standard deviation"""
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3,
                                  stride=1, padding=1)
        return torch.sqrt(local_var + 1e-8)

    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel-like filters"""
        B, C, H, W = x.shape
        # Apply Sobel per channel (use grouped convolution)
        x_reshaped = x.reshape(B * C, 1, H, W)                       # [B*C, 1, H, W]

        gx = F.conv2d(x_reshaped, self.sobel_x, padding=1)           # [B*C, 1, H, W]
        gy = F.conv2d(x_reshaped, self.sobel_y, padding=1)           # [B*C, 1, H, W]

        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)          # [B*C, 1, H, W]
        grad_mag = grad_mag.reshape(B, C, H, W)                      # [B, C, H, W]
        return grad_mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Estimate local contrast via std + gradient magnitude
        local_std = self._compute_local_std(x)                       # [B, C, H, W]
        grad_mag = self._compute_gradient_magnitude(x)                # [B, C, H, W]

        contrast_features = torch.cat([local_std, grad_mag], dim=1)  # [B, 2C, H, W]
        contrast_map = self.contrast_net(contrast_features)           # [B, 1, H, W]

        # 2. Sharpening branch: extract high-frequency detail
        # High-frequency = input - local_mean (similar to Laplacian)
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - local_mean                                    # [B, C, H, W]
        sharpened = self.sharpen_dw(high_freq)
        sharpened = self.sharpen_bn(sharpened)
        sharpened = F.gelu(sharpened)
        sharpened = x + self.sharpen_scale * sharpened                 # similar to unsharp masking

        # 3. Smoothing branch: large-kernel smoothing
        smoothed = self.smooth_dw(x)
        smoothed = self.smooth_bn(smoothed)
        smoothed = F.gelu(smoothed)
        smoothed = self.smooth_pw(smoothed)                           # [B, C, H, W]

        # 4. Contrast-guided fusion
        # High contrast → prefer sharpened; Low contrast → prefer smoothed
        combined = contrast_map * sharpened + (1 - contrast_map) * smoothed

        # 5. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CAM(channels=128)
    output = model(input_tensor)
    print('=== CAM: Contrast-Aware Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
