import os
import shutil
import numpy as np
import random
import torch
import pandas as pd
import sys
sys.path.append("/home/qir21001/AAAI2022/models")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models import CNN,LSTM,dilation_CNN,FCN,MLP
import torch.nn.functional as F
import logging
import torch,torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import seaborn as sns
from datetime import datetime
import itertools
def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def save_checkpoint(state, is_best, filename='checkpoint.pkl'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pkl')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
def generate_negatives(data_negatives,n_negatives):
    data_n = []
    for i in range(n_negatives):
        data = data_negatives[:,i,:]
        data_n.append(data)
    data_n = torch.cat(data_n,dim=0)
    return data_n

    
    
def aug_data(data_a, data_p,aug_positives):
    '''Return the augmented data'''
    data_p = data_p.cpu().numpy()
    data_a = data_a.cpu().numpy()
    batch_size = data_a.shape[0]
    shape = data_a.shape[1]
    k_1 = np.power((data_a-data_p),2) 
    r = np.expand_dims(np.sqrt(np.sum(k_1,axis=1)),axis=1)
    for i in range(aug_positives):
        v = np.random.randint(-2, 2, size=(batch_size, shape))
        length = np.expand_dims(np.sqrt(np.sum(np.square(v),axis=1)),axis=1)
        e = random.random()
        v = v/length
        v = v*e*r
        v = v + data_a                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        data_p = np.append(data_p,v,axis=0)
    
    data_p = torch.tensor(data_p) # [6 * batchsize,timestep]
   
    # print("Mixed_x.shape",mixed_x.shape)
    
    # print("y_a.shape",y_a.shape)
    return data_p

 # get the index of mined negatvies samples for each batch of positive samples   
def triplet_index(data_a,data_p,data_n,n_negatives,aug_positives):
    batchsize = data_a.shape[0]
    triplet_index = []
    if(len(data_a.shape)>2):
        data_a = data_a.reshape(data_a.shape[0],-1)
        data_p = data_p.reshape(data_p.shape[0],-1)
        data_n = data_n.reshape(data_n.shape[0],-1)
    #cos_degree = []
    for i in range(1+aug_positives):
        t = data_p[i*batchsize:(i+1)*batchsize,:]
        value = torch.cosine_similarity(data_a,data_p[i*batchsize:(i+1)*batchsize,:],dim=1) # batchsize * timestep
        index_p = []
        #cos_degree = []
        for j in range(n_negatives):
            cos_sim = torch.cosine_similarity(data_a,data_n[j*batchsize:(j+1)*batchsize,:],dim=1)
        
            #cos_degree.append(cos_sim)
            # batchsize * 1 
            #cos = [t.cpu().numpy() for t in cos_sim]
            """for i in cos:
                t = math.acos(i)
                print("acos----",t)
                k = math.degrees(t)
                print("degree---",k)
            """
            #theta_1 = [math.degrees(math.acos(i)) for i in cos]# get the degree of each pair of Van & Vap
            index = torch.where(cos_sim>=value,1,0) # (batchsize +1)* 1, get where is the negative samples we need 
            index_p.append(index)        
        triplet_index.append(index_p)
    #cos_degree = torch.cat(cos_degree,dim=0)
    #degree = torch.rad2deg(torch.acos(cos_degree))
    #print("minimum degree:%f,largest degree:%f"%(torch.min(degree),torch.max(degree)))
    return triplet_index

def mixup(data_n,number,n_negatives,batchsize,alpha=1):
    lam_list = []
    data_list = []
    for i in range(n_negatives):
        data_list.append(data_n[i*batchsize:(i+1)*batchsize])
        lam_list.append(alpha)
    lam_list = tuple(lam_list)
    data_list = torch.stack(data_list,0)
    for i in range(number):
        lam = np.random.dirichlet(lam_list)
       #SSS
        data = torch.stack([i*j for i,j in zip(data_list,lam)],0)
        data = torch.sum(data,0)
        data_n = torch.cat((data_n,data),0)
    return data_n 

def plot_TSNE(data,data_label,file_path,nclasses):
    if(len(data.shape)>2):
        data = data.reshape(data.shape[0],-1)
    if not isinstance(data_label, np.ndarray):
        data_label = data_label.detach().cpu().numpy()
    if not isinstance(data,np.ndarray):
        data = data.detach().cpu().numpy()
    print(np.isnan(data))
    print(np.isfinite(data))
    tsne = TSNE(n_components=2,init='pca',random_state=42).fit_transform(data)
    x_min,x_max = tsne.min(0),tsne.max(0)
    x_norm = (tsne-x_min)/(x_max - x_min)
    df = pd.DataFrame()
    df['y'] = data_label
    '''
    markers = ['o', '^']
    #all_markers = list(itertools.product(markers, repeat=10))
   
    
    df['com_1'] = x_norm[:,0]
    df['com_2'] = x_norm[:,1]
    fig = sns.scatterplot(x='com_1',y='com_2',hue = df.y.tolist(),palette = sns.color_palette('hls',nclasses),data=df)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(file_path+'.jpg',dpi=400) 
    
'''
    #basic_markers = ['o', '^', 's', 'D', 'v' , '*']

# Generate a cyclic iterator of basic markers
    #cyclic_markers = itertools.cycle(basic_markers)
    
# Take as many markers as there are unique classes
    #unique_classes = np.unique(df['y'])
   # markers_for_classes = list(itertools.islice(cyclic_markers, len(unique_classes)))

# Create a dictionary mapping classes to markers
    #class_marker_mapping = dict(zip(unique_classes, markers_for_classes))

# Add a new column 'marker' based on the class_marker_mapping
    #df['marker'] = df['y'].map(class_marker_mapping)

# Create scatter plot with different colors and shapes
    
    
    #fig = sns.scatterplot(x='com_1', y='com_2', hue='y', style='marker',palette=sns.color_palette('hls', nclasses),markers=basic_markers,data=df)
             
    
   # fig = sns.scatterplot(x='com_1', y='com_2', hue='y', style='y',
                         # palette=sns.color_palette('hls', nclasses),
                          #markers=markers, data=df)
    #scatter_fig = fig.get_figure()
    #scatter_fig.savefig(file_path+'.jpg',dpi=400) 
    
   # color_list = ['b','g','r','c','k','grey','orange','tomato','peru']

    print(data_label.shape)
    plt.figure(figsize=(6,6))
   
        #marker = class_marker_mapping[data_label[i]]
        
    plt.scatter(x_norm[:,0],x_norm[:,1],marker='o',c=data_label,cmap='coolwarm',s=60)
   #     sc.set_facecolor("none")
  
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.show()
    plt.savefig(file_path+'.png',dpi=400)
    
  
def define_model(model_name,input_dim):
    if(model_name == 'resnet'):
        model = torchvision.models.resnet18(pretrained=False, num_classes=128)
        model.conv1 = nn.Conv2d(1,64,kernel_size=1,stride=1,padding=0,bias=False) 
        # dim_mlp = model.fc.in_features 
        # model.fc = nn.Linear(dim_mlp, nclasses)
       
    if(model_name=='LSTM'):
        model = LSTM.LSTM(input_dim=input_dim, hidden_dim=64, layer_dim=1, output_dim=128)
        # model.fc = nn.Sequential(nn.Linear(64, 128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,nclasses))
        
    if(model_name =='FCN'):
        model = FCN.FCN(in_channels=input_dim)
        # model.fc =nn.Linear(128,nclasses)
        
    if(model_name =='dilation_CNN'):
        model = dilation_CNN.dilation_CNN(in_channels=input_dim)
    if(model_name == 'CNN'):
        model = CNN.CNNmodel(nclasses=128)
    if(model_name =='ts2vec'):
        model = tsencoder.TSEncoder(320, input_dims=input_dim, hidden_dims=64, depth=10)
    return model


def generate_pos(data, label):
    device = data.device
    new_data = torch.zeros_like(data, device=device)
    mask_dict = {}

    for i in torch.unique(label):
        mask = (label == i).nonzero().flatten()
        mask_dict[i.item()] = mask
    
    cnt = 0
    label_indices = label.long()

   # ca
    for i in label_indices:
       
        for key, value in mask_dict.items():
            if i == key:
                same_class_indices = value
                same_class_indices = same_class_indices[same_class_indices!=cnt]
                if len(same_class_indices) > 0:
                    selected_index = torch.randperm(same_class_indices.shape[0], device=device)[0].to(device)
                    new_data[cnt] = data[same_class_indices[selected_index]]
                else:
                    new_data[cnt] = data[cnt]
                break
        cnt += 1 
            
    return new_data[:cnt]             
