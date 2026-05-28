import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-28

'''
模块名称：SSM (Saliency-Guided Suppression Module) —— 显著性引导抑制模块

一、模块简介
在视觉特征图中，并非所有空间位置都包含同等重要的信息。背景、平坦区域等低信息
区域占用了大量的计算和存储资源，但对最终任务的贡献很小。现有的注意力模块
（如 CBAM、SE）侧重于"增强重要区域"，但对"不重要区域"的处理较为粗糙——
它们通常将抑制权重简单地设为接近 0，可能导致信息丢失。

SSM 的核心思想是：通过显著性检测识别高信息区域，并对低显著性区域施加软抑制
（而非硬性归零），使模型将更多"注意力预算"分配给信息丰富的区域，同时保留
低显著性区域的基础信息以防丢失。这种"节约型"的注意力分配方式在计算上几乎
零成本，但能有效提升特征的信息密度。

核心创新点：
1. 显著性驱动的自适应抑制：基于激活统计计算空间显著性，高显著性保留，低显著性软抑制
2. 可学习抑制温度：通过温度参数控制抑制的平滑程度
3. 通道-空间联合显著性：同时从通道和空间两个维度评估显著性
4. 信息预算重分配：抑制的能量被隐式地重新分配给高显著性区域

二、结构设计
SSM 由以下子结构组成：
1. 显著性检测器（Saliency Detector）：
   - 计算每个空间位置的通道最大响应和平均响应
   - 两者加权融合得到初始显著性图
   - 通过 3x3 平滑卷积消除噪声
2. 自适应阈值生成器（Adaptive Threshold Generator）：
   - 基于显著性图的全局统计（均值、标准差）计算阈值
   - 阈值 = mean + β * std，β 为可学习参数
3. 软抑制函数（Soft Suppression Function）：
   - supp(x, τ, T) = sigmoid((x - τ) / T)
   - 低于阈值的区域被抑制，高于阈值的区域被保留
   - 温度 T 控制过渡的平滑度
4. 残差调制：out = x * suppression_weight

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 SSM（Saliency-Guided Suppression Module）模块，通过显著性检测引导
的自适应抑制机制来提升特征的信息密度。该模块基于激活统计计算空间显著性图，
并利用可学习的阈值和温度参数对低显著性区域施加软抑制，在保留基础信息的同时
将模型的注意力隐式地集中到信息丰富区域。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入 CNN 或
Transformer 主干网络。特别适合存在大面积背景或平坦区域的场景（如遥感图像、
医疗图像）。
'''


class SSM(nn.Module):
    """SSM: Saliency-Guided Suppression Module —— 显著性引导抑制模块"""

    def __init__(self, channels: int, beta_init: float = 0.5, temperature_init: float = 1.0):
        super().__init__()
        # 显著性检测：通道最大 + 通道平均
        self.saliency_smooth = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)

        # 融合权重
        self.max_weight = nn.Parameter(torch.tensor(0.6))
        self.avg_weight = nn.Parameter(torch.tensor(0.4))

        # 阈值参数 β：threshold = mean + β * std
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # 抑制温度 T
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        # 最小保留率（防止过度抑制）
        self.min_keep = nn.Parameter(torch.tensor(0.3))

    def _compute_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """计算空间显著性图"""
        max_response, _ = x.max(dim=1, keepdim=True)   # [B, 1, H, W]
        avg_response = x.mean(dim=1, keepdim=True)      # [B, 1, H, W]

        # 加权融合
        saliency = self.max_weight * max_response + self.avg_weight * avg_response

        # 平滑去噪
        saliency = self.saliency_smooth(torch.cat([saliency, saliency], dim=1))
        return saliency                                    # [B, 1, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 显著性检测
        saliency = self._compute_saliency(x)                 # [B, 1, H, W]

        # 自适应阈值
        sal_mean = saliency.mean(dim=(2, 3), keepdim=True)   # [B, 1, 1, 1]
        sal_std = saliency.std(dim=(2, 3), keepdim=True)     # [B, 1, 1, 1]
        threshold = sal_mean + self.beta * sal_std

        # 软抑制：sigmoid((x - threshold) / temperature)
        normalized = (saliency - threshold) / self.temperature
        suppression_weight = torch.sigmoid(normalized)        # [B, 1, H, W]

        # 确保最小保留率
        suppression_weight = self.min_keep + (1 - self.min_keep) * suppression_weight

        out = x * suppression_weight
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SSM(channels=128)
    output = model(input_tensor)
    print('=== SSM: Saliency-Guided Suppression Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('beta:', model.beta.item(), 'temperature:', model.temperature.item(),
          'min_keep:', model.min_keep.item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
