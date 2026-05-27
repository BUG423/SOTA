import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：AFM (Adaptive Frequency Modulation) —— 自适应频率调制模块

一、模块简介
传统卷积操作本质上是空间域的局部特征提取，缺乏对频域信息的显式建模。图像中的
不同频率成分对应不同层次的视觉信息——低频对应颜色和平坦区域，中频对应纹理，
高频对应边缘和噪声。现有方法（如 JPEG 压缩、频域学习）通常在整个图像级别处理
频率，忽视了不同空间位置对不同频率成分的需求差异。

AFM 的核心思想是：使用不同核大小的并行卷积来近似不同频率带的滤波器，并通过
频率选择器为每个空间位置自适应地调制各频率成分的重要性，增强判别性频率成分。

核心创新点：
1. 多频带分解：使用 kernel_size = 1, 3, 5, 7 的深度可分离卷积近似不同频率带通滤波器
2. 空间自适应频率调制：每个位置学习其最优的频率成分组合
3. 频率重要性学习：通过全局统计学习各频带的整体重要性
4. 高效的频率重组：从多频带特征重组成增强的单一特征图

二、结构设计
AFM 由以下子结构组成：
1. 多频带分析器（Multi-band Analyzer）：
   - 4 个并行深度可分离卷积分支，kernel_size 分别为 1, 3, 5, 7
   - 小核捕获高频细节，大核捕获低频结构
2. 频率重要性预测器（Frequency Importance Predictor）：
   - 全局池化 → FC → 4 维输出，预测各频带的全局重要性
3. 空间调制器（Spatial Modulator）：
   - 对输入特征使用 1x1 卷积生成 4 通道的空间调制权重
4. 频率重组：使用全局+空间权重对各频带特征加权融合

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 AFM（Adaptive Frequency Modulation）模块，用于在空间域内实现自适应
的频率成分调制。该模块通过多尺度卷积分支近似不同频率带的滤波器，并利用频率
选择器为每个空间位置动态调整各频率成分的贡献，从而增强特征的频率多样性。"

四、适用任务
适用于图像分类、目标检测、图像恢复等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。对纹理丰富和细节敏感的任务尤为有效。
'''


class AFM(nn.Module):
    """AFM: Adaptive Frequency Modulation —— 自适应频率调制模块"""

    kernel_sizes = [1, 3, 5, 7]

    def __init__(self, channels: int):
        super().__init__()
        self.num_bands = len(self.kernel_sizes)
        # 多频带分析分支
        self.bands = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            for k in self.kernel_sizes
        ])
        # 频率重要性预测器
        self.freq_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, self.num_bands, bias=False),
            nn.Sigmoid(),
        )
        # 空间调制器
        self.spatial_modulator = nn.Sequential(
            nn.Conv2d(channels, self.num_bands, kernel_size=1, bias=False),
        )
        # 输出投影
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 多频带输出
        band_outs = [band(x) for band in self.bands]           # N × [B, C, H, W]
        stacked = torch.stack(band_outs, dim=1)                  # [B, N, C, H, W]

        # 全局频率重要性 [B, N, 1, 1]
        global_w = self.freq_importance(x).view(B, self.num_bands, 1, 1, 1)

        # 空间调制权重 [B, N, 1, H, W]
        spatial_w = self.spatial_modulator(x).unsqueeze(2)       # [B, N, 1, H, W]
        spatial_w = torch.softmax(spatial_w, dim=1)

        # 融合权重：全局 × 空间
        weights = global_w * spatial_w                           # [B, N, 1, H, W]

        # 加权融合
        out = (stacked * weights).sum(dim=1)                     # [B, C, H, W]
        out = self.output_proj(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = AFM(channels=128)
    output = model(input_tensor)
    print('=== AFM: Adaptive Frequency Modulation ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
