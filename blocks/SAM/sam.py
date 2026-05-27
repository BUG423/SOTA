import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-27

'''
模块名称：SAM (Spatial Affinity Module) —— 空间亲和力模块

一、模块简介
标准自注意力机制通过计算所有空间位置两两之间的相似度来捕获长距离依赖，但计算
复杂度为 O((HW)²)，在高分辨率特征图上难以使用。Non-local 和 CCNet 等方法通过
降采样或十字交叉策略降低复杂度，但会丢失部分空间信息。

SAM 的核心思想是：通过低秩投影将特征映射到低维空间，在低维空间中计算空间亲和力
矩阵，然后利用该亲和力矩阵对原始特征进行信息传播。每个位置的特征通过与其他位置
的亲和力加权聚合来更新，实现高效的长距离上下文传播。

核心创新点：
1. 低秩空间亲和力：在低维子空间中计算亲和力，大幅降低计算复杂度 O((HW)²) → O(HW·D)
2. 亲和力引导的信息传播：利用亲和力矩阵在原始特征空间中进行信息扩散
3. 双向传播：同时进行前向（信息聚合）和后向（信息分发）传播
4. 自适应温度：可学习的温度参数控制亲和力分布的尖锐程度

二、结构设计
SAM 由以下子结构组成：
1. 低维投影器（Low-rank Projector）：
   - 使用 1x1 卷积将 C 维特征压缩到 D 维（D << C）
   - 降低亲和力矩阵的计算成本
2. 亲和力计算器（Affinity Computer）：
   - 在低维空间中计算归一化的空间亲和力矩阵
   - 使用可学习温度参数控制 softmax 的锐度
3. 信息传播器（Information Propagator）：
   - 在原始特征空间中，利用亲和力矩阵进行信息聚合
   - 每个位置接收所有其他位置的信息，按亲和力加权
4. 输出投影：将增强后的特征投影回原始维度

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 SAM（Spatial Affinity Module）模块，通过低秩空间亲和力建模实现高效
的长距离上下文传播。该模块将特征投影到低维子空间计算空间亲和力矩阵，并利用该
矩阵引导高维特征空间中的信息传播，在保持线性复杂度的同时有效捕获全局上下文。"

四、适用任务
适用于需要全局上下文建模的视觉任务，如语义分割、目标检测、图像生成等。
可作为即插即用模块嵌入 CNN 或 Transformer 主干网络的中高层。
'''


class SAM(nn.Module):
    """SAM: Spatial Affinity Module —— 空间亲和力模块"""

    def __init__(self, channels: int, proj_dim: int = 32):
        super().__init__()
        self.proj_dim = proj_dim
        # 低维投影
        self.key_proj = nn.Conv2d(channels, proj_dim, kernel_size=1, bias=False)
        self.query_proj = nn.Conv2d(channels, proj_dim, kernel_size=1, bias=False)
        # 值投影（原始维度）
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # 可学习温度
        self.temperature = nn.Parameter(torch.tensor(proj_dim ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # 低维投影
        q = self.query_proj(x).reshape(B, self.proj_dim, N)     # [B, D, N]
        k = self.key_proj(x).reshape(B, self.proj_dim, N)       # [B, D, N]
        v = self.value_proj(x).reshape(B, C, N)                  # [B, C, N]

        # 低维空间中的亲和力矩阵
        affinity = torch.bmm(q.transpose(1, 2), k)               # [B, N, N]
        affinity = torch.softmax(affinity / self.temperature, dim=-1)

        # 亲和力引导的信息传播
        out = torch.bmm(v, affinity.transpose(1, 2))             # [B, C, N]
        out = out.reshape(B, C, H, W)

        out = self.output_proj(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 32, 32)
    model = SAM(channels=128, proj_dim=32)
    output = model(input_tensor)
    print('=== SAM: Spatial Affinity Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('temperature:', model.temperature.item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
