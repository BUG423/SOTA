import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-30

'''
模块名称：DSM (Dual-Scale Modulator) —— 双尺度调制器

一、模块简介
在视觉任务中，特征的多尺度表示对处理不同大小的目标至关重要。FPN、ASPP
等方法通过并行多分支结构获取多尺度特征，但它们通常采用"先分后合"的策略，
各尺度分支之间缺少在中间计算阶段的深度交互，导致粗尺度特征缺乏细节信息、
细尺度特征缺乏语义上下文。

DSM 的核心思想是：构建"粗-细"双尺度互调结构。粗尺度分支通过下采样和
大核卷积获取大感受野的全局语义信息；细尺度分支保持原始分辨率，通过标准
3x3 卷积提取局部细节。关键创新在于双尺度互调机制——粗尺度特征通过空间
注意力为细尺度特征提供"在哪里看"的上下文指导；细尺度特征通过残差注入
为粗尺度特征补充"丢失了什么"的细节信息。这种双向调制使得粗尺度和细尺度
特征互相补充，而非简单拼接。

核心创新点：
1. 双尺度互调机制：粗-细两个尺度在计算过程中互相调制，而非孤立计算
2. 上下文引导：粗尺度为细尺度提供空间注意力图，指示语义重要区域
3. 细节回注：细尺度高频残差注入粗尺度分支，补偿下采样丢失的细节
4. 自适应尺度融合：每个通道独立学习两个尺度的融合权重

二、结构设计
DSM 由以下子结构组成：
1. 粗尺度分支（Coarse Branch）：
   - 2x 自适应平均池化下采样
   - 5x5 逐通道卷积（大感受野）
   - 1x1 逐点卷积进行通道混合
   - 双线性插值上采样恢复分辨率
2. 细尺度分支（Fine Branch）：
   - 3x3 逐通道卷积（局部细节）
   - 1x1 逐点卷积进行通道混合
   - 保持原始分辨率
3. 粗→细引导（Coarse-to-Fine Guidance）：
   - 粗尺度特征通过 1x1 卷积 + Sigmoid 生成空间注意力图
   - 空间注意力图作用于细尺度特征，引导细节关注
4. 细→粗注入（Fine-to-Coarse Injection）：
   - 细尺度与粗尺度（上采样后）的残差
   - 通过可学习的注入强度注入粗尺度分支
5. 自适应融合：
   - 两个分支输出通过可学习权重加权组合
   - 残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 DSM（Dual-Scale Modulator）模块，通过粗-细双尺度互调机制
增强多尺度特征表示。该模块包含一个粗尺度分支和一个细尺度分支：粗尺度
分支通过大感受野捕获全局语义上下文，细尺度分支保持高分辨率局部细节。
两个分支之间通过上下文引导和细节回注实现双向调制——粗尺度为细尺度提供
空间注意力先验，细尺度为粗尺度补偿下采样丢失的高频细节。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要同时处理大目标和小目标的场景，
如多尺度目标检测、高分辨率分割等。
'''


class DSM(nn.Module):
    """DSM: Dual-Scale Modulator —— 双尺度调制器"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # --- Coarse branch (large receptive field via downsampling) ---
        self.coarse_dw = nn.Conv2d(channels, channels, 5, padding=2,
                                    groups=channels, bias=False)
        self.coarse_bn = nn.BatchNorm2d(channels)
        self.coarse_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # --- Fine branch (preserves resolution, local details) ---
        self.fine_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.fine_bn = nn.BatchNorm2d(channels)
        self.fine_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # --- Coarse-to-Fine guidance: spatial attention from coarse ---
        self.c2f = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # --- Fine-to-Coarse injection: detail compensation ---
        self.f2c_inject = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # --- Output refinement ---
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Learnable per-channel fusion weights
        self.scale_w = nn.Parameter(torch.ones(2, channels, 1, 1) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # --- Coarse branch: downsample → large kernel → upsample ---
        x_down = F.adaptive_avg_pool2d(x, (max(2, H // 2), max(2, W // 2)))
        coarse = self.coarse_dw(x_down)
        coarse = self.coarse_bn(coarse)
        coarse = F.gelu(coarse)
        coarse = self.coarse_pw(coarse)                              # [B, C, H/2, W/2]
        coarse = F.interpolate(coarse, size=(H, W), mode='bilinear',
                               align_corners=False)                   # [B, C, H, W]

        # --- Fine branch: original resolution, local details ---
        fine = self.fine_dw(x)
        fine = self.fine_bn(fine)
        fine = F.gelu(fine)
        fine = self.fine_pw(fine)                                    # [B, C, H, W]

        # --- Coarse-to-Fine guidance ---
        attn_map = self.c2f(coarse)                                   # [B, C, H, W]
        fine_guided = fine * attn_map                                 # [B, C, H, W]

        # --- Fine-to-Coarse injection: detail residual ---
        detail = fine_guided - coarse                                 # high-freq residual
        coarse_refined = coarse + self.f2c_inject.tanh() * detail     # controlled injection

        # --- Adaptive scale fusion ---
        w = F.softmax(self.scale_w, dim=0)                            # [2, C, 1, 1]
        fused = w[0] * coarse_refined + w[1] * fine_guided            # [B, C, H, W]

        # --- Refine and residual ---
        out = self.refine(fused)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = DSM(channels=128)
    output = model(input_tensor)
    print('=== DSM: Dual-Scale Modulator ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    w = F.softmax(model.scale_w, dim=0)
    print('scale_weights (coarse, fine):', w.mean(dim=(1, 2, 3)).detach().tolist())
    print('f2c_inject_strength:', model.f2c_inject.tanh().mean().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
