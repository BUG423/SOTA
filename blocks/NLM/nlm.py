import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：NLM (Non-local Modulation Module) —— 非局部调制模块

一、模块简介
经典的非局部网络（Non-local Network）通过计算所有位置对之间的亲和力
来聚合全局信息，但其输出是对所有位置特征的加权平均——这种方式本质上是
一个"平滑"操作，在捕获长程依赖的同时也模糊了局部细节。此外，标准非局部
块对所有位置同等对待，缺少对"哪些位置需要非局部信息、哪些位置依赖局部
信息已足够"的精细判断。

NLM 的核心思想是：将非局部操作从"聚合"转变为"调制"——使用非局部亲和力
矩阵生成空间自适应的调制信号，而非直接聚合特征。具体而言，NLM 计算
查询（Query）与键（Key）之间的亲和力，但不使用该亲和力直接加权值（Value），
而是将其转化为一个调制图（modulation map），用于逐位置、逐通道地调节
原始特征。这种方式既利用了长程依赖进行信息调制，又保留了原始特征的
局部结构和细节精度。

核心创新点：
1. 非局部调制而非聚合：用长程亲和力调制局部特征，避免过度平滑
2. 调制强度自适应：每个位置学习从长程上下文中获取多少调制信息
3. 残差调制：调制信号以残差方式作用，保留原始特征的完整性
4. 高效投影：低秩 QKV 投影降低计算复杂度

二、结构设计
NLM 由以下子结构组成：
1. QKV 投影：
   - Q：1x1 卷积投影到降维空间
   - K：1x1 卷积投影到降维空间 + 空间池化降低分辨率
   - V：仅用于信息提取，1x1 卷积投影
2. 亲和力计算：
   - Q 与 K 的矩阵乘法得到亲和力图
   - Softmax 归一化
3. 调制信号生成：
   - 亲和力 × V 得到聚合的上下文信息
   - 通过卷积网络将上下文转化为调制信号
   - Tanh 激活确保调制信号有正有负
4. 调制应用：
   - 输出 = 输入 + 调制强度 × 调制信号
5. 输出精炼

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 NLM（Non-local Modulation Module）模块，将非局部操作从
传统的特征聚合转变为特征调制。该模块利用全局位置亲和力生成空间自适应
的调制信号，以残差方式调节局部特征响应——在引入长程上下文信息的同时
保持局部细节的完整性，避免了标准非局部块的过度平滑问题。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要长程依赖建模但又不希望损失
空间精度的场景，如高分辨率分割、细粒度分类等。
'''


class NLM(nn.Module):
    """NLM: Non-local Modulation Module —— 非局部调制模块"""

    def __init__(self, channels: int, reduction: int = 4, pool_size: int = 16):
        super().__init__()
        inner = max(1, channels // reduction)
        self.pool_size = pool_size

        # Q projection: keep spatial resolution for fine modulation
        self.proj_q = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
        )

        # K projection: reduce spatial resolution for efficiency
        self.proj_k = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
        )

        # V projection: information to be modulated
        self.proj_v = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
        )

        # Modulation signal generator: context → modulation map
        self.modulation_generator = nn.Sequential(
            nn.Conv2d(inner, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Tanh(),
        )

        # Learnable modulation strength (per-channel)
        self.mod_strength = nn.Parameter(torch.zeros(1, channels, 1, 1))

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

        # 1. QKV projections
        Q = self.proj_q(x)                                           # [B, inner, H, W]
        K = self.proj_k(x)                                           # [B, inner, H, W]
        V = self.proj_v(x)                                           # [B, inner, H, W]

        # 2. Downsample K and V for efficiency
        ks = max(2, min(self.pool_size, H // 2))
        K_down = F.adaptive_avg_pool2d(K, (ks, ks))                  # [B, inner, ks, ks]
        V_down = F.adaptive_avg_pool2d(V, (ks, ks))                  # [B, inner, ks, ks]

        # 3. Compute affinity: Q @ K^T → normalize
        Q_flat = Q.flatten(2)                                        # [B, inner, N], N = H*W
        K_flat = K_down.flatten(2)                                   # [B, inner, M], M = ks*ks
        affinity = torch.bmm(Q_flat.transpose(1, 2), K_flat)         # [B, N, M]
        affinity = F.softmax(affinity / (inner ** 0.5), dim=-1)      # scaled softmax

        # 4. Aggregate context: affinity @ V
        V_flat = V_down.flatten(2).transpose(1, 2)                   # [B, M, inner]
        context = torch.bmm(affinity, V_flat)                        # [B, N, inner]
        context = context.transpose(1, 2).reshape(B, -1, H, W)       # [B, inner, H, W]

        # 5. Generate modulation signal from context
        modulation = self.modulation_generator(context)              # [B, C, H, W]

        # 6. Apply modulation with learnable strength
        out = x + self.mod_strength * modulation                     # [B, C, H, W]

        # 7. Refine
        out = self.refine(out)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = NLM(channels=128)
    output = model(input_tensor)
    print('=== NLM: Non-local Modulation Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('mod_strength mean:', model.mod_strength.mean().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
