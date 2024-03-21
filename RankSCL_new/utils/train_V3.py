import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn 
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import define_model,generate_pos
from loss.Ranking_loss import Ranking_loss_3
from numpy import inf
from utils.augmentation import aug_data
import torch.nn.functional as F
from models.FCN import FCN_V2,FCN
import random

def jitter(x, sigma=0.01):
    device = x.device
    return x + torch.normal(mean=0., std=sigma, size=x.shape).to(device)

'''def pos_aug(data,number,M):
    pos_index = torch.randint(0, data.size(0), (1,))
    pos = data[pos_index]
    
    if(number>0):
        data_aug = [jitter(pos,sigma=0.03) for i in range(number)]
        data_aug = torch.cat(data_aug,dim=0).to(data.device)
        data_all = torch.cat((data, data_aug), axis=0)
        return data_all
    elif(number<=0):
        return data[:M]
'''
def aug_data(data,labels):
    data_aug = [jitter(data,sigma=0.03) for i in range(1)]
    data_aug = torch.cat(data_aug,dim=0).to(data.device)
    data_all = torch.cat((data, data_aug), axis=0)
    label =  torch.cat((labels,labels),axis=0)
    return data_all, label
    
def train_model(data,model,device,encode):
    device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
   
    if encode ==True:
        h,_ = model(data.float())
        
    else:
        _,h = model(data.float())
        h = F.normalize(h,dim=(1,2))
    return h
    
def train_encoder(dataset,train_loader,model_name,device,logger,train,seed,epochs_up,M,lr=1e-5,weight_decay=1e-8,batchsize=128,distance='EU'):
    device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")
    input_dim = train.shape[1]
    model = FCN(in_channels=input_dim)
    model.to(device)
    loss_list = []
    best_loss = inf
    save_dir = '/home/qir21001/AAAI2022/result_model_V3/'+dataset+'_'+model_name+'_'+str(batchsize)+'_'+str(M)+'_'+str(lr)+'_'+str(epochs_up)+'_'+str(seed)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    
    logger.debug("-------------begin training ---------------")
    for epoch in range(epochs_up):
        
        for step,(data,labels) in enumerate(train_loader,start=0):
            
            
            data,labels = data.to(device),labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            #if(model_name == 'LSTM'):
             #   if(data.ndim==2):
              #      out= model(data.unsqueeze(2).float().to(device))
           
                
            out,_ = model(data.float().to(device))
               # out_a,_,out_p,_,out_n,_ = classifier(data_a.unsqueeze(1).float().to(device)),classifier(data_p.unsqueeze(1).float().to(device)),classifier(data_n.unsqueeze(1).float().to(device))
            
            #if(model_name =='ts2vec'):
              #  train_1, train_2,train_3 = data_a.unsqueeze(2),data_p.unsqueeze(2),data_n.unsqueeze(2)
               # out_a,out_p,out_n = model(train_1.to(device)),model(train_2.to(device)),model(train_3.to(device))
            out_norm = F.normalize(out,dim=1).to(device)
           
            #if(model_name == 'ts2vec'):
             #   loss = hierarchical_rank_loss(out_a, out_p,out_n,n_negatives,batchsize,number,distance,aug_positives,alpha=0.5, temporal_unit=0)
            #if(aug_anchor > 0):
            #     data_all,label = aug_data(out,label,aug_anchor)
            anchor_index = random.randint(0,len(out_norm)-1)
            anchor = out_norm[anchor_index]
            anchor_label = labels[anchor_index]
            data_all,label_all = aug_data(out_norm,labels)
            pos_sample = [torch.tensor(data_all[i]) for i in range(len(data_all))  if label_all[i] == anchor_label]
            neg_sample = [torch.tensor(data_all[i]) for i in range(len(data_all))  if label_all[i] != anchor_label]
            pos_sample = torch.stack(pos_sample)
            neg_sample = torch.stack(neg_sample)
            pos_data = pos_sample[:20]
            neg_data = neg_sample[:20]
            anchor_data = np.tile(anchor,(20,1))
            
           
        
           
            
            
            
            loss = Ranking_loss_3(anchor_data,pos_data,neg_data,distance,device)
            
            # validation 
           
            #print("pos_time: %.3f, aug_time:%.3f,loss_time:%.3f"%(pos_time,aug_time,loss_time))    
            loss.backward(retain_graph=False)
            optimizer.step()                                                                                                                                                                                                                                                 
        
        #if loss <=best_loss:
        #    best_loss = loss
           
       
        if(epoch%10==0):
            if 'loss' in locals():
                logger.debug("Epoch %d ----- loss %.3f"%(epoch+1,loss))
    torch.save(model.state_dict(),os.path.join(save_dir,'Rank_model.pt'))
    fig,ax = plt.subplots(figsize=(10,10))
    plt.plot(loss_list,'g--')
    plt.show(block=False)
    fig.savefig('Results/'+model_name+'_'+'Rank_add_new_model'+dataset+'_loss.jpg')
    plt.close() 
    
   # plot_TSNE(train,train_labels,file_path='TSNE/'+dataset+'_Encoder_repre',nclasses=nclasses) 
    
  
        
        