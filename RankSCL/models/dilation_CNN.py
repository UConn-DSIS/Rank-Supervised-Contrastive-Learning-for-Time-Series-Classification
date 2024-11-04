import torch
import torch.nn as nn
import torch.nn.functional as F
# from numpy.lib.arraypad import pad

class dilation_CNN(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, stride=1, padding=0,dilation=1), # in_chanels :输入向量的通道，词向量的维度，out_channels:卷积产生的通道，多少个一维卷积，kernel_size:卷积核尺寸，stride:卷积核步长，padding：输入每一层边补充0的层数，dilation:卷积核元素之间的间距
            #nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            
           # nn.Conv1d(24, 64, kernel_size=3, stride=1, padding=0,dilation=1),
            
           # nn.BatchNorm1d(64),
           # nn.Conv1d(24,32,kernel_size=3,stride=1,padding=0,dilation=1),
            #nn.ReLU(),
            #nn.BatchNorm1d(32),
            
            #nn.ReLU(),
            nn.Conv1d(24,64,kernel_size=5,stride=1,padding=0,dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0,dilation=1),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Flatten(), 
            nn.Linear(254464,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU())
    def forward(self, x):
        return self.network(x)
            # nn.MaxPool1d(3),
            # layer 4 
            #nn.Conv1d(128, 1024, kernel_size=3, stride=1, padding=0,dilation=1),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            #nn.BatchNorm1d(1024),
            # nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2), # output: 128 x 62
'''
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            '''
            # nn.MaxPool1d(2), # output: 256 x 31
            
         
        
   