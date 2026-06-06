import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：RCM (Recursive Convolution Module) —— 递归卷积模块

一、模块简介
标准卷积神经网络的每一层具有固定的处理深度——无论输入特征的复杂程度
如何，每个位置都经过相同次数的卷积变换。然而，不同空间位置的特征复杂
程度差异显著：物体边缘和纹理区域可能需要更多次的非线性变换来精炼特征，
而平坦区域可能一次变换就已足够。这种"一刀切"的深度策略导致计算资源
的浪费——简单区域被过度处理，复杂区域可能处理不足。

RCM 的核心思想是：使用权重共享的递归卷积实现"自适应深度"的特征处理。
具体而言，RCM 在同一个模块内对输入进行多次递归卷积——每次迭代使用相同
的卷积权重（权重共享），但通过一个"终止门"（termination gate）在每
个空间位置独立地决定是否继续递归。复杂区域可以经历更多次迭代，简单
区域则提前退出。这种设计在保持参数高效（权重共享）的同时实现了空间
自适应的处理深度。

核心创新点：
1. 递归卷积处理：相同权重多次应用，逐步精炼特征
2. 空间自适应深度：逐位置学习最优的递归迭代次数
3. 软终止机制：连续的门控值实现平滑的深度过渡
4. 权重共享效率：递归共享参数，参数量不随迭代次数增长

二、结构设计
RCM 由以下子结构组成：
1. 递归卷积单元（Recursive Convolution Unit）：
   - 3x3 逐通道卷积 + 1x1 逐点卷积 + GELU
   - 权重在迭代间共享
2. 终止门网络（Termination Gate Network）：
   - 以当前累积特征为输入
   - 1x1 压缩 → GELU → 1x1 → Sigmoid
   - 输出 [B, 1, H, W] 的终止概率
3. 累积器（Accumulator）：
   - 维护迭代间的累积特征
   - 每次迭代以门控方式混合新特征和累积特征
4. 最终融合：
   - 所有迭代步的特征通过可学习权重融合
5. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 RCM（Recursive Convolution Module）模块，通过权重共享的
递归卷积实现空间自适应的处理深度。该模块在每次递归迭代中使用相同的
卷积参数，通过终止门网络逐位置评估是否需要继续递归——纹理复杂区域
经历更多次精炼，平坦区域提前终止。权重共享策略确保参数量不随最大
迭代次数增长，而自适应深度机制使计算资源向复杂区域倾斜。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合特征复杂程度空间变化大的场景，
以及需要在参数效率和表达能力之间取得平衡的轻量化网络设计。
'''


class RCM(nn.Module):
    """RCM: Recursive Convolution Module —— 递归卷积模块"""

    def __init__(self, channels: int, reduction: int = 4, max_iters: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.max_iters = max_iters

        # Recursive convolution unit (weight-shared across iterations)
        self.rec_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.rec_bn = nn.BatchNorm2d(channels)
        self.rec_pw = nn.Conv2d(channels, channels, 1, bias=False)

        # Termination gate network
        self.term_gate = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # Per-iteration learnable contribution weights
        self.iter_weights = nn.Parameter(torch.ones(max_iters, 1, 1, 1) / max_iters)

        # Step embedding to distinguish iterations
        self.step_embeddings = nn.Parameter(
            torch.randn(max_iters, channels, 1, 1) * 0.02
        )

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

        # Initialize
        h = x                                                        # current feature state
        acc = torch.zeros_like(x)                                    # accumulated output
        alive = torch.ones(B, 1, H, W, device=x.device)              # positions still "active"
        outputs = []                                                  # collect per-iteration outputs

        for t in range(self.max_iters):
            # Add step embedding to distinguish iteration
            h_with_step = h + self.step_embeddings[t]                # [B, C, H, W]

            # Recursive convolution
            h_new = self.rec_dw(h_with_step)
            h_new = self.rec_bn(h_new)
            h_new = F.gelu(h_new)
            h_new = self.rec_pw(h_new)                               # [B, C, H, W]

            # Residual update with gate
            h = h + h_new                                             # [B, C, H, W]

            # Termination gate: probability of stopping at this step
            term_prob = self.term_gate(h)                             # [B, 1, H, W]

            # Accumulate: active positions contribute to output
            contribution = alive * term_prob * h                     # [B, C, H, W]
            acc = acc + contribution                                  # [B, C, H, W]

            # Update alive mask: positions that haven't terminated yet
            alive = alive * (1 - term_prob)                           # [B, 1, H, W]

            # Store this iteration's output for fusion
            outputs.append(h)

        # Handle positions that never terminated: use final state
        acc = acc + alive * h                                        # [B, C, H, W]

        # Weighted fusion of all iteration outputs
        fused = torch.zeros_like(x)
        for t, out_t in enumerate(outputs):
            fused = fused + self.iter_weights[t] * out_t

        # Mix accumulated and fused outputs
        combined = 0.5 * acc + 0.5 * fused                           # [B, C, H, W]

        # Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = RCM(channels=128, max_iters=3)
    output = model(input_tensor)
    print('=== RCM: Recursive Convolution Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('max_iters:', model.max_iters)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
