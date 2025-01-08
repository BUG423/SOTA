import math
import torch
import torch.nn as nn
#对比驱动特征聚合模块(CDFA):接收来自语义信息解耦模块的前景和背景特征，指导多级特征融合和关键特征增强，进一步区分待分割实体。
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),nn.BatchNorm1d(out_c))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
class CDFAPreprocess(nn.Module):
    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))
    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x
if __name__ == '__main__':
    test = CDFAPreprocess(6,2,1)# 实例化
    input = torch.randn(1, 6, 200)# 创建一个随机输入张量，形状为[Batch, Channel, Input length]
    output = test(input)# 执行前向传播
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)