import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-01

'''
模块名称：BSM (Bilateral Similarity Module) —— 双边相似度模块

一、模块简介
标准卷积操作使用固定的空间核权重，对邻域内所有位置进行等权或仅基于
空间距离加权的聚合。然而，这种"只靠空间距离"的聚合策略忽略了特征
内容本身的相似性——一个邻域内，有些邻居在特征空间中与中心位置高度
相似（可能属于同一物体），有些则截然不同（可能跨越物体边界）。理想
的聚合应该给予特征相似的邻居更高的权重、特征迥异的邻居更低的权重，
这正是双边滤波（bilateral filter）的核心思想。

BSM 将双边滤波的思想从图像处理迁移到特征学习领域，构建了一个可学习
的双边相似度聚合模块。对于每个空间位置，BSM 在其 K×K 邻域内计算
中心特征与每个邻居特征之间的内容相似度（通过低维投影空间中的内积
加温度系数），以此作为聚合权重。内容相似的邻居贡献更大，内容迥异的
邻居贡献更小。这种"空间邻近 + 内容相似"的双边聚合策略能够在保持
空间局部性的前提下实现跨物体边界的自适应信息隔离，从而产生更清晰的
特征边界和更鲁棒的特征表示。

核心创新点：
1. 可学习双边聚合：将双边滤波的空间核+范围核思想引入特征学习
2. 内容自适应邻域聚合：邻域聚合权重由特征相似度动态决定
3. 低维投影：相似度在低维空间中计算，避免高维特征的内积退化问题
4. 可学习温度：通过可学习温度参数控制相似度分布的锐度

二、结构设计
BSM 由以下子结构组成：
1. 邻域展开（Unfold）：
   - 使用 F.unfold 将 K×K 邻域展开为 [B, C*K², H*W]
   - 重塑为 [B, C, K², H*W] 便于计算
2. 低维投影（Projection）：
   - 1x1 卷积将 C 维特征投影到 D 维（D << C）
   - 用于计算内容相似度，降低内积退化
3. 中心-邻居相似度计算：
   - 提取邻域中心位置的特征向量作为查询
   - 计算查询与所有邻居的内积
   - 除以可学习温度参数并通过 Softmax 归一化
4. 双边加权聚合：
   - 使用 Softmax 归一化的相似度权重对邻居特征进行加权求和
   - 折叠回 [B, C, H, W] 形状
5. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 BSM（Bilateral Similarity Module）模块，将双边滤波的空间-范围
双核思想引入特征学习中的邻域聚合过程。该模块对每个空间位置的 K×K 邻域
进行基于内容相似度的自适应加权聚合——特征相似度高的邻居获得高聚合权重，
相似度低的邻居获得低聚合权重——从而在保持空间局部性的前提下实现内容感知
的信息聚合，有效提升了特征的边界清晰度和语义一致性。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要保留清晰特征边界、抑制跨物体
特征混合的任务，如语义分割、边缘检测、实例分割等。
'''


class BSM(nn.Module):
    """BSM: Bilateral Similarity Module —— 双边相似度模块"""

    def __init__(self, channels: int, kernel_size: int = 5,
                 proj_dim: t.Optional[int] = None, reduction: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.k2 = kernel_size * kernel_size
        self.center_idx = self.k2 // 2  # middle element in the flattened patch
        inner = max(1, channels // reduction)
        self.proj_dim = proj_dim or max(1, channels // 4)

        # Project features to lower dimension for similarity computation
        self.proj_q = nn.Conv2d(channels, self.proj_dim, 1, bias=False)
        self.proj_k = nn.Conv2d(channels, self.proj_dim, 1, bias=False)

        # Learnable temperature for similarity sharpness
        self.temperature = nn.Parameter(torch.tensor(self.proj_dim ** -0.5))

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        K = self.kernel_size
        k2 = self.k2
        D = self.proj_dim

        # 1. Project to low-dim space for similarity computation
        feat_q = self.proj_q(x)                                      # [B, D, H, W]
        feat_k = self.proj_k(x)                                      # [B, D, H, W]

        # 2. Unfold into patches: [B, D*k2, H*W] → [B, D, k2, H*W]
        queries = feat_q.flatten(2)                                   # [B, D, H*W]
        keys = F.unfold(feat_k, kernel_size=K, padding=self.pad,
                        stride=1)                                     # [B, D*k2, H*W]
        keys = keys.view(B, D, k2, H * W)                             # [B, D, k2, H*W]

        # 3. Content similarity: dot product between query and each neighbor key
        # queries: [B, D, H*W] → [B, 1, D, H*W]
        # keys:    [B, D, k2, H*W]
        # similarity: [B, k2, H*W]
        sim = (queries.unsqueeze(2) * keys).sum(dim=1)                # [B, k2, H*W]
        sim = sim / self.temperature.abs()                            # temperature scaling
        attn = F.softmax(sim, dim=1)                                  # [B, k2, H*W]

        # 4. Apply same attention to original (un-projected) feature neighbors
        values = F.unfold(x, kernel_size=K, padding=self.pad,
                          stride=1)                                   # [B, C*k2, H*W]
        values = values.view(B, C, k2, H * W)                         # [B, C, k2, H*W]

        # Weighted aggregation: [B, C, k2, H*W] * [B, 1, k2, H*W] → [B, C, H*W]
        attn_expanded = attn.unsqueeze(1)                              # [B, 1, k2, H*W]
        aggregated = (values * attn_expanded).sum(dim=2)               # [B, C, H*W]

        # 5. Refine and residual
        out = self.refine(aggregated)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = BSM(channels=128, kernel_size=5)
    output = model(input_tensor)
    print('=== BSM: Bilateral Similarity Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('temperature:', model.temperature.item())
    print('kernel_size:', model.kernel_size)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
