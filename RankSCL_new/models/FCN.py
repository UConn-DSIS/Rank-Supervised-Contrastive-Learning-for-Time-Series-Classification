import torch
import torch.nn as nn
import torch.nn.functional as F
# from numpy.lib.arraypad import pad

class FCN(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=6,dilation=2), # in_chanels :输入向量的通道，词向量的维度，out_channels:卷积产生的通道，多少个一维卷积，kernel_size:卷积核尺寸，stride:卷积核步长，padding：输入每一层边补充0的层数，dilation:卷积核元素之间的间距
            #nn.BatchNorm1d(24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            
           # nn.Conv1d(24, 64, kernel_size=3, stride=1, padding=0,dilation=1),
            
           # nn.BatchNorm1d(64),
           # nn.Conv1d(24,32,kernel_size=3,stride=1,padding=0,dilation=1),
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            
            #nn.ReLU(),
            nn.Conv1d(24,64,kernel_size=5,padding=8,dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 320, kernel_size=3,padding=8,dilation=8),
            nn.BatchNorm1d(320),
            nn.ReLU(),
 
           
            # nn.MaxPool1d(3),
            # layer 4 
            #nn.Conv1d(128, 1024, kernel_size=3, stride=1, padding=0,dilation=1),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            #nn.BatchNorm1d(1024),
            # nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2), # output: 128 x 6
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten())
        self.proj_head = nn.Sequential(nn.Linear(320,320),
                                       nn.BatchNorm1d(320),
                                       nn.ReLU(),
                                       nn.Linear(320,320))
         
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.proj_head(h)
        return out,h

class FCN_V2(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=8, padding=4), # in_chanels :输入向量的通道，词向量的维度，out_channels:卷积产生的通道，多少个一维卷积，kernel_size:卷积核尺寸，stride:卷积核步长，padding：输入每一层边补充0的层数，dilation:卷积核元素之间的间距
            #nn.BatchNorm1d(24),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128,256,kernel_size=5,padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Conv1d(256, 128, kernel_size=3,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
 
           
            # nn.MaxPool1d(3),
            # layer 4 
            #nn.Conv1d(128, 1024, kernel_size=3, stride=1, padding=0,dilation=1),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            #nn.BatchNorm1d(1024),
            # nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2), # output: 128 x 6
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten())
        self.proj_head = nn.Sequential(nn.Linear(128,128),
                                       nn.BatchNorm1d(128),
                                       nn.ReLU(),
                                       nn.Linear(128,128))
         
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.proj_head(h)
        return out,h