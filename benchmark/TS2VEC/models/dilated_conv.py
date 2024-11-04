import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
        # nn.Conv1d是PyTorch中的一个1维卷积层，用于处理1维的输入数据，例如时间序列数据或文本数据。
        # in_channels：输入张量的通道数（也就是每个时间步的特征数）。out_channels：输出张量的通道数（也就是卷积核的数量，每个卷积核会产生一个特征图）。
        #kernel_size：卷积核的大小，可以是一个整数，表示使用相同的卷积核大小，也可以是一个元组，表示每个维度上的卷积核大小。
        #stride：卷积核的步长，可以是一个整数，表示使用相同的步长，也可以是一个元组，表示每个维度上的步长。
        #padding：在输入的每个边界周围添加0的层数。
        #dilation：卷积核的膨胀率，也就是卷积核中每个元素之间的间隔。
        #groups：输入和输出的通道被分成的组数，可以用来实现分组卷积。
        #bias：是否添加偏置。
        #padding_mode：边界填充方式，可以是'zeros'或'circular'
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size): # in_channels=64, channels =[64 64 64 64 64 64 64 64 64 64 320],kernel_size=3
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
