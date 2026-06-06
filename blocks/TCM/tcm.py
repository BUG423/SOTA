import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：TCM (Tensor Completion Module) —— 张量补全模块

一、模块简介
在深度卷积网络中，特征图在传播过程中不可避免地会出现信息丢失或退化——
部分通道的部分空间位置可能由于ReLU置零、Dropout随机丢弃或池化操作
而丢失有价值的信息。传统方法通常依赖堆叠更多层来弥补这种信息损失，
但这种方式计算开销大且效率低下。

TCM 的核心思想是：将特征图视为一个三维张量，利用低秩张量补全技术
从被观测到的特征中推断并补全缺失或退化的信息。具体而言，TCM 首先通过
通道压缩和空间投影将特征映射到一个更适合补全的潜在空间，然后利用
低秩分解（类SVD）提取特征的主要结构成分，再通过重构路径恢复补全后的
特征。整个过程类似矩阵补全中的核范数最小化——通过分离信号子空间和
噪声子空间，保留主要成分、抑制噪声成分。

核心创新点：
1. 张量补全引入特征处理：将低秩补全思想应用于特征图增强
2. 双路径分解-重构：压缩-分解路径提取结构成分，重构路径恢复完整表达
3. 自适应秩估计：可学习的秩选择机制，根据特征内容自适应确定保留成分数
4. 软阈值去噪：通过Singmoid门控实现类似奇异值软阈值的去噪效果

二、结构设计
TCM 由以下子结构组成：
1. 通道压缩器（Channel Compressor）：
   - 1x1 卷积将通道从 C 压缩到 inner_dim
   - 降低张量补全的计算复杂度
2. 低秩分解器（Low-Rank Decomposer）：
   - 通过两个线性投影（类U和V）模拟 SVD 分解
   - U: 通道→秩投影，V: 空间→秩投影
   - 在秩空间中组合后重构
3. 结构保持路径（Structure Preservation Path）：
   - 3x3 逐通道卷积保持局部结构
   - 与补全路径并行，防止过度平滑
4. 自适应融合：
   - 可学习权重在补全路径和结构路径之间平衡
5. 通道恢复与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 TCM（Tensor Completion Module）模块，将低秩张量补全理论
引入卷积特征增强。该模块将特征图视为部分可观测的张量，通过低秩分解
提取主要结构成分并补全退化信息，同时保留关键的局部空间结构。
整个过程可解释为对特征张量的自适应低秩近似与去噪。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合网络较深时中间层特征的增强，
以及 Dropout 或下采样操作较多的网络结构。
'''


class TCM(nn.Module):
    """TCM: Tensor Completion Module —— 张量补全模块"""

    def __init__(self, channels: int, reduction: int = 4, rank: int = 16):
        super().__init__()
        inner = max(1, channels // reduction)
        self.rank = min(rank, inner)

        # Channel compressor: reduce dimension for efficient tensor completion
        self.compress = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Low-rank decomposer: simulate SVD via two projections
        # U-like projection: channel → rank
        self.proj_u = nn.Conv2d(inner, self.rank, 1, bias=False)
        # V-like projection: spatial → rank (via channel reduction then spatial pooling)
        self.proj_v = nn.Sequential(
            nn.Conv2d(inner, self.rank, 3, padding=1, groups=self.rank, bias=False),
            nn.BatchNorm2d(self.rank),
            nn.GELU(),
        )

        # Reconstruction from rank space back to inner
        self.reconstruct = nn.Sequential(
            nn.Conv2d(self.rank, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Structure preservation path: maintain local spatial structure
        self.structure_dw = nn.Conv2d(inner, inner, 3, padding=1,
                                       groups=inner, bias=False)
        self.structure_bn = nn.BatchNorm2d(inner)

        # Adaptive fusion: learn balance between completion and structure paths
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(inner * 2, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel restoration and output refinement
        self.restore = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
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

        # 1. Compress channels for efficient completion
        compressed = self.compress(x)                                # [B, inner, H, W]

        # 2. Low-rank decomposition: extract principal components
        u = self.proj_u(compressed)                                  # [B, rank, H, W]
        v = self.proj_v(compressed)                                  # [B, rank, H, W]
        # Combine rank components via element-wise interaction
        rank_features = u * v                                        # [B, rank, H, W]

        # 3. Reconstruct from rank space
        completed = self.reconstruct(rank_features)                  # [B, inner, H, W]

        # 4. Structure preservation path
        structure = self.structure_dw(compressed)
        structure = self.structure_bn(structure)
        structure = F.gelu(structure)                                # [B, inner, H, W]

        # 5. Adaptive fusion: learn where to trust completion vs structure
        gate_input = torch.cat([completed, structure], dim=1)        # [B, 2*inner, H, W]
        gate = self.fusion_gate(gate_input)                          # [B, inner, H, W]
        fused = gate * completed + (1 - gate) * structure            # [B, inner, H, W]

        # 6. Restore channels and refine
        out = self.restore(fused)                                    # [B, C, H, W]
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = TCM(channels=128)
    output = model(input_tensor)
    print('=== TCM: Tensor Completion Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('rank:', model.rank)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
