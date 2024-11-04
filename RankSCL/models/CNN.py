import torch
import torch.nn as nn
import torch.nn.functional as F
# from numpy.lib.arraypad import pad

class CNNmodel(nn.Module):
    def __init__(self,nclasses):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=2, stride=1, padding=1,dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(24, 64, kernel_size=2, stride=1, padding=1,dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(3), 

            # nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(3),
            # nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(3),
            # nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2), # output: 128 x 62

            # nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2), # output: 256 x 31

            nn.Flatten(), 
            nn.Linear(14208,1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, nclasses))
        
    def forward(self, x):
        print(x.shape)
        return self.network(x)