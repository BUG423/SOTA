import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：WAM (Weighted Attention Module) —— 加权注意力模块

一、模块简介
注意力机制在计算机视觉中取得了巨大成功，但现有方法通常聚焦于单一的
注意力模式——要么是通道注意力（如 SENet），要么是空间注意力（如 CBAM），
要么是非局部注意力（如 Non-local Network）。每种注意力模式捕捉不同
类型的关系：通道注意力捕获"哪些通道重要"，空间注意力捕获"哪些位置
重要"，局部注意力捕获"邻域内的精细关系"，全局注意力捕获"长程依赖"。
然而，单一注意力模式天然存在盲区，且不同任务、不同层级对注意力模式的
需求也不同。

WAM 的核心思想是：并行计算多种注意力模式（通道注意力、空间注意力、
局部结构注意力、全局上下文注意力），然后通过学习的重要性权重对它们进行
自适应融合。具体而言，WAM 维护四个并行的注意力头（Channel Head、
Spatial Head、Local Head、Global Head），每个头从不同维度评估特征
的重要性；然后通过一个融合网络学习每个位置对四种注意力模式的依赖程度，
最终以加权和的方式组合四个注意力头的输出。

核心创新点：
1. 多模式注意力并行：同时计算通道、空间、局部、全局四种注意力
2. 注意力模式自适应融合：学习每个位置的最优注意力模式组合
3. 模式间互补：不同注意力模式覆盖不同的关系类型，形成互补
4. 低秩高效设计：各注意力头共享部分投影以减少计算开销

二、结构设计
WAM 由以下子结构组成：
1. 共享特征提取：
   - 1x1 卷积将输入映射到四个分支的共享表示空间
2. 通道注意力头（Channel Head）：
   - 全局平均池化 + 全局最大池化 → 1x1 MLP → Sigmoid
   - 输出 [B, C, 1, 1] 的通道注意力
3. 空间注意力头（Spatial Head）：
   - 通道平均 + 通道最大 → 7x7 卷积 → Sigmoid
   - 输出 [B, 1, H, W] 的空间注意力
4. 局部结构头（Local Head）：
   - 3x3 逐通道卷积 → Sigmoid
   - 输出 [B, C, H, W] 的局部结构注意力
5. 全局上下文头（Global Head）：
   - QKV 投影 → 降采样亲和力 → 上采样上下文
   - 输出 [B, C, H, W] 的全局上下文调制
6. 模式融合网络：
   - 拼接四个头的输出 → 1x1 卷积 → Softmax
   - 学习每个位置的四模式融合权重
7. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 WAM（Weighted Attention Module）模块，通过并行计算多种
注意力模式并自适应融合，克服了单一注意力模式的局限性。该模块同时计算
通道、空间、局部结构和全局上下文四种注意力，并通过学习的模式融合权重
在每个空间位置实现最优的注意力组合——不同位置可以根据其内容特点
自适应地选择最合适的注意力模式或模式组合。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合特征模式复杂、需要多种注意力
机制协同工作的场景。
'''


class WAM(nn.Module):
    """WAM: Weighted Attention Module —— 加权注意力模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Conv2d(channels, inner * 4, 1, bias=False),
            nn.BatchNorm2d(inner * 4),
            nn.GELU(),
        )

        # Channel attention head
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)
        self.ch_mlp = nn.Sequential(
            nn.Conv2d(inner * 2, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention head
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # Local structure head
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.local_bn = nn.BatchNorm2d(channels)
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Global context head (lightweight non-local)
        self.global_q = nn.Conv2d(channels, inner, 1, bias=False)
        self.global_k = nn.Conv2d(channels, inner, 1, bias=False)
        self.global_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.global_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Mode fusion network: learn per-position mode weights
        self.mode_fusion = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, inner, 1, bias=False),  # ch + local + global + spatial
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 4, 1, bias=False),
            nn.Softmax(dim=1),
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

        # 1. Shared feature extraction
        shared_feat = self.shared(x)                                 # [B, 4*inner, H, W]
        inner_feats = torch.chunk(shared_feat, 4, dim=1)             # 4 × [B, inner, H, W]

        # 2. Channel attention head
        ch_avg = self.ch_avg_pool(inner_feats[0])                    # [B, inner, 1, 1]
        ch_max = self.ch_max_pool(inner_feats[0])                    # [B, inner, 1, 1]
        ch_cat = torch.cat([ch_avg, ch_max], dim=1)                  # [B, 2*inner, 1, 1]
        ch_attn = self.ch_mlp(ch_cat)                                # [B, C, 1, 1]
        ch_out = x * ch_attn                                         # [B, C, H, W]

        # 3. Spatial attention head
        sp_avg = torch.mean(inner_feats[1], dim=1, keepdim=True)     # [B, 1, H, W]
        sp_max, _ = torch.max(inner_feats[1], dim=1, keepdim=True)   # [B, 1, H, W]
        sp_cat = torch.cat([sp_avg, sp_max], dim=1)                  # [B, 2, H, W]
        sp_attn = self.sp_conv(sp_cat)                               # [B, 1, H, W]
        sp_out = x * sp_attn                                         # [B, C, H, W]

        # 4. Local structure head
        local = self.local_dw(x)
        local = self.local_bn(local)
        local = F.gelu(local)
        local_attn = self.local_gate(local)                          # [B, C, H, W]
        local_out = x * local_attn                                   # [B, C, H, W]

        # 5. Global context head
        Q = self.global_q(x)                                         # [B, inner, H, W]
        K = self.global_k(x)                                         # [B, inner, H, W]
        V = self.global_v(x)                                         # [B, C, H, W]

        # Downsample K for efficiency
        ks = max(2, min(8, H // 4))
        K_d = F.adaptive_avg_pool2d(K, (ks, ks))                     # [B, inner, ks, ks]
        V_d = F.adaptive_avg_pool2d(V, (ks, ks))                     # [B, C, ks, ks]

        Q_f = Q.flatten(2).transpose(1, 2)                           # [B, N, inner]
        K_f = K_d.flatten(2)                                         # [B, inner, M]
        V_f = V_d.flatten(2).transpose(1, 2)                         # [B, M, C]

        affinity = F.softmax(torch.bmm(Q_f, K_f) / (inner ** 0.5), dim=-1)  # [B, N, M]
        global_ctx = torch.bmm(affinity, V_f)                        # [B, N, C]
        global_ctx = global_ctx.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        global_ctx = self.global_proj(global_ctx)
        global_out = x + global_ctx                                  # [B, C, H, W]

        # 6. Mode fusion: learn per-position weights for 4 attention modes
        # Stack attention outputs for fusion
        mode_input = torch.cat([
            ch_out,      # channel attention result
            sp_out,      # spatial attention result
            local_out,   # local structure result
        ], dim=1)                                                    # [B, 3C, H, W]
        # Also include spatial attention map as spatial guidance
        mode_weights = self.mode_fusion(
            torch.cat([mode_input, sp_attn], dim=1)                  # [B, 3C+1, H, W]
        )                                                            # [B, 4, H, W]

        # Weighted combination of 4 attention modes
        w_ch = mode_weights[:, 0:1, :, :]                           # [B, 1, H, W]
        w_sp = mode_weights[:, 1:2, :, :]                           # [B, 1, H, W]
        w_local = mode_weights[:, 2:3, :, :]                         # [B, 1, H, W]
        w_global = mode_weights[:, 3:4, :, :]                        # [B, 1, H, W]

        combined = (w_ch * ch_out + w_sp * sp_out +
                    w_local * local_out + w_global * global_out)      # [B, C, H, W]

        # 7. Refine and residual
        out = self.refine(combined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = WAM(channels=128)
    output = model(input_tensor)
    print('=== WAM: Weighted Attention Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
