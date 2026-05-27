"""
ResNet 中插入即插即用模块的示例代码。

本文件展示如何将 blocks/ 目录下的原创模块嵌入到标准 ResNet-50 骨干网络中。
所有模块均遵循即插即用原则：将模块插入每个残差块的卷积之后、残差相加之前即可。
"""
import typing as t
import torch
import torch.nn as nn

from blocks.SRM.srm import SRM
from blocks.DFA.dfa import DFA
from blocks.CIM.cim import CIM
from blocks.GFF.gff import GFF
from blocks.DRS.drs import DRS
from blocks.AFM.afm import AFM
from blocks.PFA.pfa import PFA
from blocks.SAM.sam import SAM
from blocks.CRM.crm import CRM
from blocks.LCR.lcr import LCR


ATTENTION_MAP = {
    'srm': SRM,
    'dfa': DFA,
    'cim': CIM,
    'gff': GFF,
    'drs': DRS,
    'afm': AFM,
    'pfa': PFA,
    'sam': SAM,
    'crm': CRM,
    'lcr': LCR,
}


class Bottleneck(nn.Module):
    """ResNet Bottleneck + 可选的原创注意力模块"""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: t.Optional[nn.Module] = None, attention_type: str = 'none'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # 插入原创注意力模块
        if attention_type in ATTENTION_MAP:
            self.attention = ATTENTION_MAP[attention_type](out_channels * self.expansion)
        else:
            self.attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.attention is not None:
            out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 主干网络，支持插入原创注意力模块"""

    def __init__(self, num_classes: int = 1000, attention_type: str = 'none'):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1, attention_type=attention_type)
        self.layer2 = self._make_layer(128, 4, stride=2, attention_type=attention_type)
        self.layer3 = self._make_layer(256, 6, stride=2, attention_type=attention_type)
        self.layer4 = self._make_layer(512, 3, stride=2, attention_type=attention_type)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1,
                    attention_type: str = 'none') -> nn.Sequential:
        layers = []
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample, attention_type))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, attention_type=attention_type))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print('=== ResNet-50 + 原创模块参数对比 ===')
    print(f'{"Module":>10} | {"Params (M)":>10} | {"Input -> Output":>25}')
    print('-' * 55)

    input_tensor = torch.randn(1, 3, 224, 224)
    names = ['none'] + list(ATTENTION_MAP.keys())

    for attn in names:
        m = ResNet50(num_classes=1000, attention_type=attn)
        n_params = count_parameters(m) / 1e6
        label = attn.upper() if attn != 'none' else 'Baseline'
        if attn == 'none':
            o = m(input_tensor)
            print(f'{label:>10} | {n_params:>10.2f} | {str(input_tensor.shape):>10} -> {str(o.shape):<10}')
        else:
            print(f'{label:>10} | {n_params:>10.2f} |')
