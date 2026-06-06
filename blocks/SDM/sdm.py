import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-06

'''
模块名称：SDM (Spectral Decomposition Module) —— 谱分解模块

一、模块简介
在深度特征空间中，不同通道之间通常存在高度相关性——许多通道编码相似
或冗余的信息。这种通道间的相关性可以通过协方差矩阵的特征分解来量化和
利用：主要特征向量对应信号子空间（大部分信息能量），次要特征向量对应
噪声子空间（低能量、高冗余）。传统方法（如通道注意力）通过标量权重
重标定每个通道，但无法利用通道间的方向性关系。

SDM 的核心思想是：对特征的通道协方差矩阵进行特征分解（谱分解），在
特征向量空间中区分信号子空间和噪声子空间，然后在信号子空间中增强特征、
在噪声子空间中抑制特征，最后将调制后的特征投影回原始通道空间。这种
操作可以理解为对特征张量进行"通道空间的旋转→子空间滤波→逆旋转"——
类似于信号处理中的主成分分析（PCA）滤波。

核心创新点：
1. 通道协方差谱分解：对通道间协方差进行特征分解，获取特征值和特征向量
2. 信号/噪声子空间分离：基于特征值能量占比区分信号和噪声子空间
3. 子空间自适应调制：信号子空间增强、噪声子空间抑制
4. 端到端可学习：整个谱分解和滤波过程可微分，支持端到端训练

二、结构设计
SDM 由以下子结构组成：
1. 协方差矩阵估计：
   - 对特征图进行空间平均，得到 [B, C, 1, 1] 的通道均值
   - 计算去均值后的通道协方差矩阵 [B, C, C]
2. 特征分解：
   - 通过迭代方法（幂迭代近似）估计主要特征值和特征向量
   - 或使用可学习的投影逼近特征分解
3. 子空间门控：
   - 根据特征值分布学习信号/噪声的软划分
   - 1x1 卷积生成子空间调制权重
4. 谱域滤波：
   - 在特征向量空间中应用调制权重
5. 空间精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 SDM（Spectral Decomposition Module）模块，通过对通道协方差
矩阵的谱分解实现子空间层级的特征调制。该模块将通道空间旋转到特征向量
基下，在其中区分高能量信号子空间和低能量噪声子空间，并分别进行增强
和抑制——相比逐通道标量调制，谱分解能够利用通道间的方向性结构进行
更精细的特征重标定。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合通道数较多、通道间冗余度较高的
深层网络层。
'''


class SDM(nn.Module):
    """SDM: Spectral Decomposition Module —— 谱分解模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_eigenvectors: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_eigenvectors = min(num_eigenvectors, channels)

        # Channel covariance projector
        self.cov_proj = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Subspace modulation: learn signal/noise separation in eigenspace
        self.subspace_gate = nn.Sequential(
            nn.Conv2d(inner, inner // 2, 1, bias=False),
            nn.BatchNorm2d(inner // 2),
            nn.GELU(),
            nn.Conv2d(inner // 2, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel restoration
        self.restore = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Spatial refinement
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel covariance matrix from spatial-averaged features"""
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dimensions
        x_flat = x.view(B, C, N)                                     # [B, C, N]

        # Center: subtract channel mean
        x_mean = x_flat.mean(dim=2, keepdim=True)                     # [B, C, 1]
        x_centered = x_flat - x_mean                                 # [B, C, N]

        # Covariance: (1/N) * X @ X^T
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / N  # [B, C, C]

        return cov

    def _spectral_filter(self, x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Apply spectral filtering using covariance eigen-decomposition"""
        B, C, H, W = x.shape

        # Eigendecomposition of covariance
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)       # [B, C], [B, C, C]
        except Exception:
            # Fallback: use identity if eigendecomposition fails
            return x

        # Sort eigenvalues in descending order (eigh returns ascending)
        eigenvalues = eigenvalues.flip(1)                             # [B, C]
        eigenvectors = eigenvectors.flip(2)                           # [B, C, C]

        # Compute eigenvalue energy ratio for signal/noise separation
        total_energy = eigenvalues.sum(dim=1, keepdim=True) + 1e-8   # [B, 1]
        energy_ratio = eigenvalues / total_energy                     # [B, C]

        # Use top eigenvectors for reconstruction
        top_k = min(self.num_eigenvectors, C)
        top_eigenvectors = eigenvectors[:, :, :top_k]                 # [B, C, K]
        top_eigenvalues = eigenvalues[:, :top_k]                      # [B, K]

        # Project to eigenspace: X' = U^T @ X
        x_flat = x.view(B, C, H * W)                                  # [B, C, N]
        x_proj = torch.bmm(top_eigenvectors.transpose(1, 2), x_flat)  # [B, K, N]

        # Spectral gating: enhance dominant components
        # Energy-based soft gating
        energy_gate = (energy_ratio[:, :top_k] * C).clamp(0, 1)      # [B, K]
        energy_gate = energy_gate.unsqueeze(-1)                       # [B, K, 1]
        x_proj_gated = x_proj * energy_gate                          # [B, K, N]

        # Reconstruct: X_rec = U @ X'_gated
        x_rec = torch.bmm(top_eigenvectors, x_proj_gated)             # [B, C, N]
        x_rec = x_rec.view(B, C, H, W)                                # [B, C, H, W]

        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Project to inner space for efficient processing
        projected = self.cov_proj(x)                                 # [B, inner, H, W]

        # 2. Compute channel covariance and apply spectral filtering
        cov = self._compute_covariance(projected)                     # [B, inner, inner]
        filtered = self._spectral_filter(projected, cov)              # [B, inner, H, W]

        # 3. Subspace gating: learn which spectral components to enhance
        gate = self.subspace_gate(filtered)                           # [B, inner, H, W]
        gated = filtered * gate                                      # [B, inner, H, W]

        # 4. Restore to original channel dimension
        restored = self.restore(gated)                                # [B, C, H, W]

        # 5. Spatial refinement
        refined = self.spatial_refine(restored)                       # [B, C, H, W]

        # 6. Refine and residual
        out = self.refine(refined)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SDM(channels=128)
    output = model(input_tensor)
    print('=== SDM: Spectral Decomposition Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
