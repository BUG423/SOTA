import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-01

'''
模块名称：IRM (Information Routing Module) —— 信息路由模块

一、模块简介
标准卷积神经网络中，所有空间位置共享相同的卷积核参数。这种"统一处理"
策略虽然高效，但忽略了不同空间位置在语义内容上的差异性——平坦区域的
特征处理需求与纹理丰富区域截然不同。虽然动态卷积通过输入相关的核生成
缓解了这一问题，但核生成的参数量较大且难以扩展到多专家场景。

IRM 的核心思想是：构建"多专家 + 内容感知路由"的计算架构。模块包含
K 个并行的专家分支（每个专家是一个轻量的深度可分离卷积块），以及一个
路由网络。路由网络根据每个空间位置的特征内容，为该位置生成 K 个专家
的混合权重——在纹理复杂区域可能路由到细节增强专家，在平坦区域路由到
平滑去噪专家，在边缘区域路由到边缘保持专家。每个位置的最终输出是各
专家输出的加权组合，权重由路由网络以内容感知的方式动态决定。

核心创新点：
1. 内容感知路由：每个空间位置根据自身特征内容动态选择专家组合
2. 轻量多专家设计：每个专家采用深度可分离卷积，参数量极低
3. 软路由策略：Softmax 归一化的连续权重，实现平滑的专家过渡
4. 空间粒度路由：路由决策在空间维度上逐位置进行，粒度最细

二、结构设计
IRM 由以下子结构组成：
1. 专家分支组（Expert Branches）：
   - K 个并行的深度可分离卷积块
   - 每个块：3x3 DWConv → BN → GELU → 1x1 PWConv
   - 不同专家在训练中自然分化为不同的处理策略
2. 路由网络（Router）：
   - 输入特征 → 1x1 卷积压缩到 inner 维 → BN → GELU → 1x1 卷积到 K 维
   - Softmax 归一化得到每个位置的 K-路路由权重 [B, K, H, W]
3. 路由聚合：
   - 将各专家输出按其对应的路由权重加权求和
4. 输出精炼：
   - 1x1 卷积 + BN 精炼融合结果
   - 残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 IRM（Information Routing Module）模块，通过多专家内容感知路由
机制实现空间自适应的特征处理。该模块包含 K 个轻量级专家分支和一个路由
网络：路由网络根据每个空间位置的特征内容为其分配专家混合权重，使不同
语义区域能够被最适合的专家组合处理，从而在保持计算效率的前提下实现
内容自适应的特征增强。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要空间自适应处理的场景，如包含
异质区域的高分辨率图像、场景理解等。
'''


class IRM(nn.Module):
    """IRM: Information Routing Module —— 信息路由模块"""

    def __init__(self, channels: int, num_experts: int = 4, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_experts = num_experts

        # K expert branches: lightweight depthwise separable conv blocks
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1, bias=False),
            )
            self.experts.append(expert)

        # Router network: input features → per-position expert mixing weights
        self.router = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, num_experts, 1, bias=False),
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

        # 1. Compute per-position routing weights
        route_logits = self.router(x)                                # [B, K, H, W]
        route_w = F.softmax(route_logits, dim=1)                     # normalize over experts

        # 2. Execute all expert branches in parallel
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))                          # each [B, C, H, W]

        # 3. Weighted combination of expert outputs
        combined = torch.zeros_like(x)
        for k in range(self.num_experts):
            weight = route_w[:, k:k+1, :, :]                         # [B, 1, H, W]
            combined = combined + weight * expert_outputs[k]          # [B, C, H, W]

        # 4. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = IRM(channels=128, num_experts=4)
    output = model(input_tensor)
    print('=== IRM: Information Routing Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('num_experts:', model.num_experts)
    # Check routing diversity
    route_w = F.softmax(model.router(input_tensor), dim=1)
    print('routing entropy (higher=more diverse):',
          (-route_w * (route_w + 1e-8).log()).sum(dim=1).mean().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
