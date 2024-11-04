from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        conv1_out_len = ((configs.seq_len + 2 * (configs.kernel_size//2) - configs.kernel_size) / configs.stride) + 1
        pool1_out_len = ((conv1_out_len+ 2*(1) - 2)/2)+ 1 
        conv2_out_len = ((pool1_out_len + 2*4-8)/1)+1
        pool2_out_len = ((conv2_out_len + 2*1 - 2 )/2)+1
        conv3_out_len = ((pool2_out_len + 2*4 - 8)/1)+1
        pool3_out_len = ((conv3_out_len + 2 * 1 - 2)/2)+1

        self.logits = nn.Linear(320 * int(pool3_out_len), configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        
        x = self.conv_block2(x)
        
        x = self.conv_block3(x)
        model_output_dim = x.shape[2]
       
        x_flat = x.reshape(x.shape[0], -1)
        
        logits = self.logits(x_flat)
        return logits, x
