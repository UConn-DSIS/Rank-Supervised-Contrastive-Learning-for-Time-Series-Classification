import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler


class LSTM(nn.Module):
 
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim,
                            num_layers = layer_dim,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim,output_dim),nn.ReLU())
        #self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim),nn.ReLU(),nn.Linear(output_dim,nclasses),nn.ReLU())
        self.batch_size = None
        self.hidden = None
        self.input_dim = input_dim
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        # print(out.size())
        out = self.fc(out[:, -1, :])
        # print(out.size())
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


        # try the 1 layer/2 
        # 把hidden_state＋attention（加weight）fully layer[+ attention]
        # pick up the hidden_state all steps + fully_layer
        
        # ranking loss augmentation 
         


