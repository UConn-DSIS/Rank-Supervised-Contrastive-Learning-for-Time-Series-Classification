import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn 
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import define_model,generate_pos
from loss.Ranking_loss import Ranking_loss_kmeans
from numpy import inf
from utils.augmentation import aug_data
import torch.nn.functional as F
from models.FCN import FCN_V2,FCN


def jitter(x, sigma=0.01):
    device = x.device
    return x + torch.normal(mean=0., std=sigma, size=x.shape).to(device)

def pos_aug(data,number):
    
    pos_data = {}
    pos_data[0] = data
    for i in range(1,number+1):
        pos_data[i] = jitter(data,sigma=0.03)
    
    return pos_data
    

        
    
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
    
def train_encoder(dataset,train_loader,model_name,device,logger,train,train_labels,seed,epochs_up,aug_positives,lr=1e-5,weight_decay=1e-8,batchsize=128,distance='EU'):
    device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")
    input_dim = train.shape[1]
    model = FCN(in_channels=input_dim)
    model.to(device)
    loss_list = []
    best_loss = inf
    save_dir = '/home/qir21001/AAAI2022/result_model_kmeans/'+dataset+'_'+model_name+'_'+str(batchsize)+'_'+str(aug_positives)+'_'+str(lr)+'_'+str(epochs_up)+'_'+str(seed)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    
    logger.debug("-------------begin training ---------------")
    for epoch in range(epochs_up):
        
        for step,(anchor,positive,negative) in enumerate(train_loader,start=0):
            
            #print(f"Ancho_shape:{anchor.size} Pos_shape:{positive.size} Neg_shape:{negative.size}")
            number = anchor.shape[1]
            batchsize = anchor.shape[0]
            anchor = anchor.reshape(anchor.shape[1]*anchor.shape[0],anchor.shape[-2],anchor.shape[-1])
            positive = positive.reshape(positive.shape[1]*positive.shape[0],positive.shape[-2],positive.shape[-1])
            negative = negative.reshape(negative.shape[1]*negative.shape[0],negative.shape[-2],negative.shape[-1])
            anchor,positive,negative = anchor.to(device), positive.to(device), negative.to(device)
            model.train()
            optimizer.zero_grad()
            
            #if(model_name == 'LSTM'):
             #   if(data.ndim==2):
              #      out= model(data.unsqueeze(2).float().to(device))
           
                
            z_a, _ = model(anchor.float().to(device))
            z_p, _ = model(positive.float().to(device))
            z_n, _ = model(negative.float().to(device))
               # out_a,_,out_p,_,out_n,_ = classifier(data_a.unsqueeze(1).float().to(device)),classifier(data_p.unsqueeze(1).float().to(device)),classifier(data_n.unsqueeze(1).float().to(device))
            
            #if(model_name =='ts2vec'):
              #  train_1, train_2,train_3 = data_a.unsqueeze(2),data_p.unsqueeze(2),data_n.unsqueeze(2)
               # out_a,out_p,out_n = model(train_1.to(device)),model(train_2.to(device)),model(train_3.to(device))
            a_norm = F.normalize(z_a,dim=1).to(device)
            p_norm = F.normalize(z_p,dim=1).to(device)
            n_norm = F.normalize(z_n,dim=1).to(device)
           
            #if(model_name == 'ts2vec'):
             #   loss = hierarchical_rank_loss(out_a, out_p,out_n,n_negatives,batchsize,number,distance,aug_positives,alpha=0.5, temporal_unit=0)
            #if(aug_anchor > 0):
            #     data_all,label = aug_data(out,label,aug_anchor)
          
            
            
            pos_all = pos_aug(p_norm,aug_positives)
             
            loss = Ranking_loss_kmeans(a_norm,pos_all,n_norm,distance,aug_positives,batchsize,number,device)
            
            # validation 
         #   loss = torch.tensor(loss,dtype=torch.float,requires_grad=True)
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
    
  
        
        