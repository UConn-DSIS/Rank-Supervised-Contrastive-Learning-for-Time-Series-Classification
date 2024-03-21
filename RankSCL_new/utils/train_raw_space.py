import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn 
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import define_model,generate_pos
from loss.Ranking_loss import Ranking_loss
from numpy import inf
from utils.augmentation import aug_data
import torch.nn.functional as F
import time
def train_model(data,model_name,model,device,encode):
    device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    if(model_name == 'CNN'):
        if(data.ndim==2):
            h = model(data.unsqueeze(1).float())
        else:
            h = model(data.float())
    elif(model_name == 'LSTM'):
        data = data.permute(0,2,1)
        h = model(data.float())
        #h = F.normzalize(h,dim=1)
        
    elif(model_name =='FCN'):
        if encode ==True:
            if(data.ndim==2):
                h,_ = model(data.unsqueeze(1).float())
            else:
                h,_ = model(data.float())
        else:
            if(data.ndim==2):
                _,h = model(data.unsqueeze(1).float())
                h = F.normalize(h,dim=1)
            else:
                _,h = model(data.float())
                h = F.normalize(h,dim=(1,2))
    elif(model_name =='resnet'):
        data_1 = data.unsqueeze(1)
        h =model(data_1.float())
        
    elif(model_name == 'dilation_CNN'):
        if(data.ndim==2):
            h = model(data.unsqueeze(1).float())
        else:
            h = model(data.float())
    #elif(model_name == 'ts2vec'):
     #   h = model(data.unsqueeze(2).float())
    return h
    
def train_encoder(dataset,train_loader,model_name,device,logger,train,train_labels,nclasses,seed,epochs_up,loss,aug_positives=5,lr=1e-5,weight_decay=1e-8,batchsize=128,distance='EU'):
    device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")
    #define the encdoer model
    input_dim = train.shape[1]
    model = define_model(model_name,input_dim)
    model.to(device)
    
    
    loss_list = []
    best_loss = inf
    save_dir = '/home/qir21001/AAAI2022/Abla_results/'+dataset+'_rawspace'+model_name+'_'+str(batchsize)+'_'+str(aug_positives)+'_'+str(lr)+'_'+str(epochs_up)+'_'+str(seed)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    
    logger.debug("-------------begin training ---------------")
    for epoch in range(epochs_up):
        
        for step,(data,labels) in enumerate(train_loader,start=0):
            
            
            data,labels = data.to(device),labels.to(device)
            pos = generate_pos(data,labels)
            pos_all,label = aug_data(pos,labels,aug_positives) 
            data_all = pos_all
            model.train()
            optimizer.zero_grad()
            
            if(model_name == 'CNN'):
                if(data.ndim==2):
                    out= model(data_all.unsqueeze(1).float().to(device))
                else: 
                    out = model(data_all.float())
            if(model_name =='FCN'):
                
                out,_ = model(data_all)
               # out_a,_,out_p,_,out_n,_ = classifier(data_a.unsqueeze(1).float().to(device)),classifier(data_p.unsqueeze(1).float().to(device)),classifier(data_n.unsqueeze(1).float().to(device))
            if(model_name == 'dilation_CNN'):
                
                
                out = model(data_all)
            if(model_name =='resnet'):
                
                train = data_all.unsqueeze(1)
              
                out = model(train.to(device))
            #if(model_name =='ts2vec'):
              #  train_1, train_2,train_3 = data_a.unsqueeze(2),data_p.unsqueeze(2),data_n.unsqueeze(2)
               # out_a,out_p,out_n = model(train_1.to(device)),model(train_2.to(device)),model(train_3.to(device))
                
           
            #if(model_name == 'ts2vec'):
             #   loss = hierarchical_rank_loss(out_a, out_p,out_n,n_negatives,batchsize,number,distance,aug_positives,alpha=0.5, temporal_unit=0)
            #if(aug_anchor > 0):
            #     data_all,label = aug_data(out,label,aug_anchor)
            if(model_name =='LSTM'):
                
                train = data_all.permute(0,2,1)
              
                out = model(train.to(device))
            
            
            
            data_out = F.normalize(out,dim=1)
            if loss == 'rank':
                loss = Ranking_loss(data_out,label,distance,device)
                loss.backward(retain_graph=False)
            elif loss == 'ce':
                criterion = torch.nn.CrossEntropyLoss() 
                loss = criterion(torch.tensor(data_out,requires_grad=True),torch.tensor(label,dtype=torch.long))
                loss.backward(retain_graph=True)
            # validation 
           
            #print("pos_time: %.3f, aug_time:%.3f,loss_time:%.3f"%(pos_time,aug_time,loss_time))    
            
            optimizer.step()                                                                                                                                                                                                                                                 
        
        #if loss <=best_loss:
        #    best_loss = loss
           
       
        if(epoch%10==0):
            
            
            logger.debug("Epoch %d ----- loss %.3f"%(epoch+1,loss))
    torch.save(model.state_dict(),os.path.join(save_dir,'Rank_model.pt'))
    #fig,ax = plt.subplots(figsize=(10,10))
    #plt.plot(loss_list,'g--')
    #plt.show(block=False)
    #fig.savefig('Results/'+model_name+'_'+'Rank_add_new_model'+dataset+'_loss.jpg')
    #plt.close() 
    
   # plot_TSNE(train,train_labels,file_path='TSNE/'+dataset+'_Encoder_repre',nclasses=nclasses) 
    
  
        
        