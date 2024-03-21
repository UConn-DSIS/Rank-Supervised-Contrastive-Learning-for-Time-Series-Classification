import numpy as np
import torch
import torch.nn.functional as F

def jitter(x, sigma=0.01):
    device = x.device
    return x + torch.normal(mean=0., std=sigma, size=x.shape).to(device)

def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def rotation(e,dim,axis):
   
    cos = torch.cos(e)
    sin =  torch.sin(e)
    sub_mat = torch.ger(axis,axis)*(1-cos)
    sub_mat += torch.diag(torch.repeat_interleave(cos,dim))
    cross_mat = torch.zeros((dim,dim))
    cross_mat[0,1] = -axis[2]
    cross_mat[0,2] = axis[1]
    cross_mat[1,0] = axis[2]
    cross_mat[1,2] = -axis[0]
    cross_mat[2,0] = -axis[1]
    cross_mat[2,1] = axis[0]
    cross_mat *= sin
    sub_mat += cross_mat 
    R = sub_mat - torch.eye(dim)
    R = R.T @ R
    R = R/torch.max(torch.abs(torch.eig(R)[0]))
    
    
    return R
    
# do the augmentation for the anchor on the embedding space
def aug_data(data,label,number):
   
    device = data.device
    data_norm = F.normalize(data,dim=1)
    data_all = data_norm.to(device)
    # generate the random degree
    label_1 = label.repeat(number*2+1).to(data_norm.device)
    if(number>0):
        # “cuda”  cuda kernel
        data_1 = [jitter(data,sigma=0.03) for i in range(number)]
        data_1 = torch.cat(data_1,dim=0).to(device)
        data_2 = [jitter(data,sigma=0.05) for i in range(number)]
        data_2 = torch.cat(data_2,dim=0).to(device)
        data_all = torch.cat((data_all, data_1,data_2), axis=0)
    return data_all,label_1
#    for i in range(number):
'''
        e = np.random.uniform(0,math.pi/18)
        e = torch.tensor(e,dtype=torch.float32)
        axis = torch.randn(data.shape[0]) 
        axis /= torch.norm(axis)
        R = rotation(e,data.shape[0],axis)
        R = R.to(data.device)
        R_1 = rotation(-e,data.shape[0],axis)
        R_1 = R_1.to(data.device)
        data_1 = torch.matmul(R,data)
        data_2 = torch.matmul(R_1,data)
'''
 #       data_1 = jitter(data,sigma=0.03)
  #      data_2 = jitter(data,sigma=0.05)
        
    
    # torch.repeat replace toch.cat jitter
    
    
    
    
