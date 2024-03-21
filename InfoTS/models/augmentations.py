import random
from models.augclass import *
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch.nn.modules.module import Module

class AUGs(Module):
    def __init__(self,caugs,p=0.2):
        super(AUGs,self).__init__()

        self.augs = caugs
        self.p = p
    def forward(self,x_torch):
        x = x_torch.clone()
        for a in self.augs:
            if random.random()<self.p:
                x = a(x)
        return x.clone(),x_torch.clone()

class RandomAUGs(Module):
    def __init__(self,caugs,p=0.2):
        super(RandomAUGs,self).__init__()

        self.augs = caugs
        self.p = p
    def forward(self,x_torch):
        x = x_torch.clone()

        if random.random()<self.p:
            x =random.choice(self.augs)(x)
        return x.clone(),x_torch.clone()

class AutoAUG(Module):
    def __init__(self, aug_p1=0.2, aug_p2 = 0.0, used_augs=None, device=None, dtype=None) -> None:
        super(AutoAUG,self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        all_augs = [subsequence(),cutout(), jitter(), scaling(), time_warp(), window_slice(), window_warp(),cutout()]
        #all_augs = [subsequence(0.01),subsequence(0.3),subsequence(0.5),subsequence(0.7),subsequence(0.99),jitter(0.01),jitter(0.1),jitter(0.3),jitter(1.0),jitter(3.0)]
#         all_augs = [subsequence(0.01),subsequence(0.02),subsequence(0.03),subsequence(0.04),subsequence(0.05),subsequence(0.06),subsequence(0.07),subsequence(0.08),subsequence(0.09),subsequence(0.1),subsequence(0.90),subsequence(0.91),subsequence(0.92),subsequence(0.93),subsequence(0.94),subsequence(0.95),subsequence(0.96),subsequence(0.97),subsequence(0.98),subsequence(0.99),jitter(0.01),jitter(0.02),jitter(0.03),jitter(0.04),jitter(0.05),jitter(0.06),jitter(0.07),jitter(0.08),jitter(0.09),jitter(0.1),jitter(3.01),jitter(3.02),jitter(3.03),jitter(3.04),jitter(3.05),jitter(3.06),jitter(3.07),jitter(3.08),jitter(3.09),jitter(3.1),subsequence(0.3),subsequence(0.5),subsequence(0.7),jitter(0.1),jitter(0.3),jitter(1.0)]


        if used_augs is not None:
            self.augs = []
            for i in range(len(used_augs)):
                if used_augs[i]:
                    self.augs.append(all_augs[i])
        else:
            self.augs = all_augs
        self.weight = Parameter(torch.empty((2,len(self.augs)), **factory_kwargs))
        self.reset_parameters()
        self.aug_p1 = aug_p1
        self.aug_p2 = aug_p2

    def get_sampling(self,temperature=1.0, bias=0.0):

        if self.training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(self.weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.cuda()
            gate_inputs = (gate_inputs + self.weight) / temperature
            # para = torch.sigmoid(gate_inputs)
            para = torch.softmax(gate_inputs,-1)
            return para
        else:
            return torch.softmax(self.weight,-1)


    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01)
    def forward(self,xt):
        x,t = xt
        if self.aug_p1 ==0 and self.aug_p2==0:
            return x.clone(), x.clone()
        para = self.get_sampling()

        if random.random()>self.aug_p1 and self.training:
            aug1 = x.clone()
        else:
            xs1_list = []
            for aug in self.augs:
                xs1_list.append(aug(x))
            xs1 = torch.stack(xs1_list, 0)
            xs1_flattern = torch.reshape(xs1, (xs1.shape[0], xs1.shape[1] * xs1.shape[2] * xs1.shape[3]))
            aug1 = torch.reshape(torch.unsqueeze(para[0], -1) * xs1_flattern, xs1.shape)
            aug1 = torch.sum(aug1,0)

        aug2 = x.clone()

        return aug1,aug2
