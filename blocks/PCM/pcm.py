import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-05-31

'''
模块名称：PCM (Phase-Coherence Module) —— 相位一致性模块

一、模块简介
在标准卷积神经网络中，所有计算均在空间域中进行。卷积操作本质上是一种
局部的、平移等变的线性变换，擅长捕获局部空间模式，但对特征的全局结构
信息和频域特性缺乏显式建模。虽然自注意力机制可以捕获长程依赖，但计算
开销大，且缺少对频域中相位（结构）和幅度（能量）的解耦分析能力。

PCM 的核心思想是：通过 2D 快速傅里叶变换（FFT）将特征从空间域变换到
频域，在频域中对幅度谱和相位谱分别进行差异化处理，然后通过逆变换回到
空间域。幅度谱编码了各频率成分的能量强度，相位谱编码了空间结构信息
（边缘、纹理的相位一致性）。PCM 对两者采用不同的处理策略——幅度谱经过
轻量级自适应重标定以抑制噪声主导的频率分量，相位谱经过空间感知的精炼
以增强结构一致性——从而实现在保留空间结构的前提下净化特征表示。

核心创新点：
1. 频域解耦：通过 FFT 将特征显式分解为幅度谱和相位谱，实现能量与结构
   的分离处理
2. 自适应幅度重标定：通过频域可学习权重对幅度谱进行自适应缩放，抑制
   噪声主导的频率分量
3. 相位一致性增强：通过处理相位的 (cos, sin) 表示，在保持周期性的
   前提下增强结构信息的相位一致性
4. 频域-空间域双路径：频域处理的全局性与空间域卷积的局部性互补

二、结构设计
PCM 由以下子结构组成：
1. 频域变换（Frequency Transform）：
   - 2D 实数 FFT（rfft2），将 [B, C, H, W] 变换为复数谱
   - 提取幅度谱（magnitude）和相位角（phase angle）
2. 幅度处理分支（Magnitude Branch）：
   - 对数变换压缩动态范围
   - 1x1 卷积 → GELU → 1x1 卷积 → Sigmoid 生成频域重标定权重
   - 将权重应用于原始幅度谱
3. 相位处理分支（Phase Branch）：
   - 将相位角转换为 (cos(θ), sin(θ)) 双通道表示
   - 深度可分离卷积进行空间感知的相位精炼
   - 转换回相位角
4. 频域重组与逆变换：
   - 处理后的幅度 × exp(j × 处理后的相位) 得到新复数谱
   - 逆实数 FFT（irfft2）回到空间域
5. 残差连接：将频域增强特征与原始空间域特征融合

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 PCM（Phase-Coherence Module）模块，通过频域解耦分析增强
卷积特征的质量。该模块利用 2D FFT 将特征从空间域变换到频域，对幅度谱
和相位谱分别进行差异化处理——幅度分支自适应重标定各频率分量的能量贡献，
相位分支增强结构信息的相位一致性——再将处理后的频域表示逆变换回空间域。
这种频域-空间域双路径设计在不引入大量参数的前提下，为卷积特征注入了
全局频域先验。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合对纹理、结构和周期性模式敏感的任务，
如纹理识别、缺陷检测、遥感图像分析等。
'''


class PCM(nn.Module):
    """PCM: Phase-Coherence Module —— 相位一致性模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Magnitude branch: frequency-domain adaptive rescaling
        self.mag_encoder = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Phase branch: spatial structure refinement in (cos, sin) space
        # Input is [B, 2C, H, W] (cos and sin for each channel)
        self.phase_dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1,
                                   groups=channels * 2, bias=False)
        self.phase_bn = nn.BatchNorm2d(channels * 2)
        self.phase_pw = nn.Conv2d(channels * 2, channels * 2, 1, bias=False)

        # Output refinement (spatial domain)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Learnable balance between frequency-enhanced and original features
        self.freq_weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Forward FFT: spatial → frequency domain
        # rfft2 returns half-spectrum (conjugate symmetry), shape [B, C, H, W//2+1]
        X_freq = torch.fft.rfft2(x, norm='ortho')                    # complex [B, C, H, W//2+1]

        # 2. Decompose into magnitude and phase
        X_mag = torch.abs(X_freq)                                     # [B, C, H, W//2+1]
        X_phase = torch.angle(X_freq)                                 # [B, C, H, W//2+1]

        # 3. Magnitude branch: adaptive frequency rescaling
        # Log-compress for stable processing
        mag_log = torch.log1p(X_mag)                                  # [B, C, H, W//2+1]
        mag_weight = self.mag_encoder(mag_log)                        # [B, C, H, W//2+1]
        mag_refined = X_mag * mag_weight                              # adaptive rescaling

        # 4. Phase branch: structure refinement via (cos, sin) representation
        phase_cos = torch.cos(X_phase)                                # [B, C, H, W//2+1]
        phase_sin = torch.sin(X_phase)                                # [B, C, H, W//2+1]
        phase_repr = torch.cat([phase_cos, phase_sin], dim=1)         # [B, 2C, H, W//2+1]

        # Depthwise separable processing in (cos, sin) space
        phase_out = self.phase_dw(phase_repr)
        phase_out = self.phase_bn(phase_out)
        phase_out = F.gelu(phase_out)
        phase_out = self.phase_pw(phase_out)                          # [B, 2C, H, W//2+1]

        # Split back and reconstruct phase
        phase_cos_out = phase_out[:, :C, :, :]                        # [B, C, H, W//2+1]
        phase_sin_out = phase_out[:, C:, :, :]                        # [B, C, H, W//2+1]
        # Normalize to unit circle
        phase_norm = torch.sqrt(phase_cos_out ** 2 + phase_sin_out ** 2 + 1e-8)
        phase_cos_out = phase_cos_out / phase_norm
        phase_sin_out = phase_sin_out / phase_norm
        phase_refined = torch.atan2(phase_sin_out, phase_cos_out)     # [B, C, H, W//2+1]

        # 5. Recompose complex spectrum and inverse FFT
        X_freq_refined = mag_refined * torch.exp(1j * phase_refined)  # complex
        x_freq = torch.fft.irfft2(X_freq_refined, s=(H, W), norm='ortho')  # [B, C, H, W]

        # 6. Learnable fusion of frequency-enhanced and original features
        alpha = self.freq_weight.sigmoid()                             # [1, C, 1, 1]
        enhanced = alpha * x_freq + (1 - alpha) * x                    # [B, C, H, W]

        # 7. Refine and residual
        out = self.refine(enhanced)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = PCM(channels=128)
    output = model(input_tensor)
    print('=== PCM: Phase-Coherence Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('freq_weight (sigmoid):', model.freq_weight.sigmoid().mean().item())
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
