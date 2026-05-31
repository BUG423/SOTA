import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-31

'''
模块名称：CGM (Conditional Gating Module) —— 条件门控模块

一、模块简介
现有的通道门控机制（如 SE、ECA）通过全局池化压缩空间信息来学习通道间
的依赖关系，但这种压缩操作丢失了空间分布信息。另一方面，空间门控机制
（如 CBAM 的空间分支）虽然保留了空间结构，但缺少对"语义基元"的全局
建模能力。两种机制的共同局限在于：门控权重的生成完全取决于当前输入
特征的统计量，缺乏对高层语义概念的显式参照。

CGM 的核心思想是：引入一组可学习的"条件原型"（conditional prototypes）
作为语义参照系，门控权重的生成不再仅依赖于输入特征本身，而是通过衡量
输入特征与这些原型的语义相似度来条件化地生成。具体而言，CGM 维护 K 个
可学习的 d 维原型向量，对于输入特征图的每个空间位置，计算其 C 维特征
向量与所有原型之间的相似度（内积），通过 softmax 得到该位置在 K 个原型
上的软分配权重；随后，这些原型分配权重被用于生成空间自适应的通道门控
信号。这种"原型条件化"机制使门控过程具有了语义参照，能更精准地选择
和增强具有特定语义模式的特征。

核心创新点：
1. 条件原型机制：引入可学习原型向量作为门控的语义参照系
2. 相似度驱动门控：门控权重由输入特征与原型的语义相似度决定，而非
   单纯的统计量
3. 软原型分配：每个空间位置通过 softmax 在所有原型上进行软分配，
   实现平滑的语义过渡
4. 紧凑原型设计：原型的维度远小于通道数，保持极低的参数开销

二、结构设计
CGM 由以下子结构组成：
1. 条件原型库（Prototype Bank）：
   - K 个可学习的 d 维原型向量（参数矩阵 [K, d]）
   - d << C（默认 d = C/4），保持紧凑
2. 特征投影（Feature Projection）：
   - 1x1 卷积将 C 维特征投影到 d 维，与原型共享同一低维空间
   - 用于计算特征与原型的相似度
3. 原型相似度计算（Prototype Similarity）：
   - 投影特征与原型矩阵的内积
   - Softmax 归一化得到每个位置的 K-路原型分配权重 [B, K, H, W]
4. 条件门控生成（Conditional Gate Generation）：
   - 将原型分配权重与可学习的原型门控嵌入结合
   - 生成 [B, C, H, W] 的空间-通道门控信号
5. 门控应用与残差：
   - Sigmoid 激活后对输入特征进行门控
   - 残差连接输出

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 CGM（Conditional Gating Module）模块，通过引入可学习条件原型
将语义参照机制融入特征门控过程。该模块维护一组紧凑的语义原型向量，门控
信号的生成以输入特征与这些原型的相似度为条件——特征与原型的语义匹配程度
决定了各位置的门控强度。这种条件化门控策略使模块能够基于高层语义概念对
特征进行选择性增强，而非简单依赖统计量驱动的启发式规则。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要语义感知特征增强的任务，如
细粒度分类、语义分割、开放词汇检测等。
'''


class CGM(nn.Module):
    """CGM: Conditional Gating Module —— 条件门控模块"""

    def __init__(self, channels: int, num_prototypes: int = 8,
                 proto_dim: t.Optional[int] = None, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_prototypes = num_prototypes
        self.proto_dim = proto_dim or max(1, channels // 4)

        # Learnable prototype bank: [K, d]
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, self.proto_dim)
                                       * (self.proto_dim ** -0.5))

        # Feature projection: C → d (shared space with prototypes)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, self.proto_dim, 1, bias=False),
            nn.BatchNorm2d(self.proto_dim),
        )

        # Prototype-specific gate embeddings: each proto has its own gating pattern
        # [K, inner] → expanded to [K, channels] via two-layer MLP
        self.proto_gate_embed = nn.Parameter(torch.randn(num_prototypes, inner)
                                              * (inner ** -0.5))

        # Gate decoder: prototype assignment → channel-spatial gate
        # Takes per-position prototype weights [B, K, H, W] and
        # prototype embeddings [K, C] to produce [B, C, H, W] gate
        self.gate_decoder = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
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
        K = self.num_prototypes
        D = self.proto_dim

        # 1. Project features to prototype space: [B, D, H, W]
        feat_proj = self.proj(x)                                      # [B, D, H, W]

        # 2. Compute similarity between each position's feature and each prototype
        # feat_proj: [B, D, H, W] → reshape to [B, H*W, D]
        feat_flat = feat_proj.flatten(2).transpose(1, 2)              # [B, H*W, D]
        # L2-normalize for cosine-like similarity
        feat_norm = F.normalize(feat_flat, p=2, dim=-1)               # [B, H*W, D]
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)         # [K, D]
        # Similarity: [B, H*W, K]
        sim = torch.matmul(feat_norm, proto_norm.t())                  # [B, H*W, K]

        # 3. Soft prototype assignment: [B, K, H, W]
        sim = sim.transpose(1, 2).view(B, K, H, W)                    # [B, K, H, W]
        # Temperature-scaled softmax for sharper/softer assignment
        assign = F.softmax(sim * (D ** 0.5), dim=1)                   # [B, K, H, W]

        # 4. Generate conditional gate from prototype assignments
        # Combine prototype embed with assignment via weighted sum
        # proto_gate_embed: [K, inner] → each proto contributes a gating pattern
        # assign: [B, K, H, W] → per-position proto weights
        # Output: [B, inner, H, W] mixed gating features
        # Matrix multiply: for each (b,h,w), weight[K] × embed[K, inner] → [inner]
        assign_flat = assign.permute(0, 2, 3, 1).reshape(B * H * W, K)  # [B*H*W, K]
        gate_feat_flat = torch.matmul(assign_flat, self.proto_gate_embed)  # [B*H*W, inner]
        gate_feat = gate_feat_flat.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, inner, H, W]

        # 5. Decode to channel-spatial gate
        gate = self.gate_decoder(gate_feat)                             # [B, C, H, W]

        # 6. Apply gate and residual
        modulated = x * gate                                            # [B, C, H, W]
        out = self.refine(modulated)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CGM(channels=128, num_prototypes=8)
    output = model(input_tensor)
    print('=== CGM: Conditional Gating Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('prototypes shape:', model.prototypes.shape)
    print('num_prototypes:', model.num_prototypes)
    print('proto_dim:', model.proto_dim)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
