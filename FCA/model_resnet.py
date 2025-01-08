"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""
import torch.nn as nn
import torch
import os,sys
from FCA import FCA
def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 1D convolution with kernel size 3 """
    return nn.Conv1d(in_planes,out_planes,kernel_size=3,stride=stride,padding=dilation,groups=groups,bias=False,dilation=dilation,)
def conv1x1(in_planes, out_planes, stride=1):
    """ 1D convolution with kernel size 1 """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    """ Supports: groups=1, dilation=1 """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes * self.expansion)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class ResNet1D(nn.Module):
    def __init__(self,block_type,in_dim,out_dim,group_sizes):
        super(ResNet1D, self).__init__()
        self.base_plane = 64
        self.inplanes = self.base_plane
        # Input module
        self.input_block = nn.Sequential(nn.Conv1d(in_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.base_plane),nn.ReLU(inplace=True),nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        # Residual groups
        self.residual_groups = nn.Sequential(
            self._make_residual_group1d(block_type, 64, group_sizes[0], stride=1),
            self._make_residual_group1d(block_type, 128, group_sizes[1], stride=2),
            self._make_residual_group1d(block_type, 256, group_sizes[2], stride=2),
            self._make_residual_group1d(block_type, 512, group_sizes[3], stride=2),)
        #FCA
        self.fca = FCA(512)
        #output module
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512,2)
    def _make_residual_group1d(self, block, planes, group_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride),nn.BatchNorm1d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, group_size):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = self.fca(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0),-1))
        return x
if __name__ == '__main__':
    test = ResNet1D(BasicBlock,6,2,[2,2,2,2])# 实例化
    input = torch.randn(1, 6, 200)# 创建一个随机输入张量，形状为[Batch, Input length, Channel]
    output = test(input)# 执行前向传播
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)
    print(test.get_num_params())