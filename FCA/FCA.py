import math
import torch
from torch import nn
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out
class FCA(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(FCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)#全局平均池化
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv1d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()
    def forward(self, input):
        x = self.avg_pool(input)#输入bcl(1,6,1)
        x1 = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)#bcl
        x2 = self.fc(x).transpose(-1, -2)#blc
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.transpose(-1, -2)).transpose(-1, -2)
        out = self.sigmoid(out)
        return input*out
if __name__ == '__main__':
    test = FCA(6)# 实例化
    input = torch.randn(1, 6, 200)# 创建一个随机输入张量，形状为[Batch, Input length, Channel]
    output = test(input)# 执行前向传播
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)

