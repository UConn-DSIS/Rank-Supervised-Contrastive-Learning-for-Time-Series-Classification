import numpy as np
import torch
import torch.nn.functional as F
import time
import random
import math
import copy
def Ranking_loss(data,label,distance,device):
  
    device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    label = label.to(device)

    
    n = label.size(0)
    label_mat = label.repeat(n,1)
    label_all = (label_mat == label_mat.T)
    # calculate distance matrix
    if distance == 'Cosine':
        distance_matrix = torch.cosine_similarity(data.unsqueeze(1), data.unsqueeze(0),dim=2)
        distance_matrix = -distance_matrix
    elif distance == 'EU':
        distance_matrix = torch.cdist(data, data, p=2)
       
   

    # calculate the sum of negative ranks for each positive sample
    sum_neg = []
    
    for i in range(distance_matrix.shape[0]):
        pos_indices = torch.nonzero(label_all[i,:])
        neg_indices = torch.nonzero(~label_all[i,:])
        for j, pos_idx in enumerate(pos_indices):
            if(pos_idx!=i):
                pos = distance_matrix[i,pos_idx].to(device)
                neg_row = distance_matrix[i,neg_indices].to(device)  # extract the corresponding rows from the distance matrix
                valid_indices = torch.where(neg_row <= pos)[0]
                neg_valid = neg_row[valid_indices].to(device)
                rank = torch.nn.Sigmoid()(pos - neg_valid).to(device)
                sum_neg.append(torch.sum(rank))
    
    # calculate the loss
    #loss = torch.mean(torch.log1p(torch.stack(sum_neg)))
    # calculate the loss
    '''
    if not sum_neg:
        sum_neg = torch.tensor([0.], dtype=torch.float32,requires_grad=True).view(-1,1)
        loss = torch.mean(torch.atan(sum_neg))
    else:
        loss = torch.mean(torch.atan(torch.stack(sum_neg)))
    '''
    loss = torch.mean(torch.atan(torch.stack(sum_neg)))
    return loss
'''
    device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    label = label.to(device)
    #labels = torch.cat([torch.arange(args.batchsize) for i in range(2)], dim=0)
    
    # normalize the emebeddings 
    #data_a = F.normalize(data_a,dim=1)
     
    #data_p = F.normalize(data_p,dim=1)
    #data_n = F.normalize(data_n,dim=1)
  
    #n_negatives = n_negatives + number
    # calculate the triplet_index
    #triplet_matrix = triplet_index(data_a,data_p,data_n,n_negatives,aug_positives) 
    # calculate the sim
    #distance_ap = []
    #distance_an = []
    if distance == 'Cosine':
        distance_matrix = torch.cosine_similarity(data,data.T,dim=1)
    if distance == 'EU':
        distance_matrix = torch.cdist(data,data.T,p=2)
    t1 = torch.nn.Sigmoid()
    mask = torch.eye(data.shape[0]).bool()
    distance_matrix = distance_matrix.mask_select(~mask)
    label_all = (label == label.T)
    label_all = label_all.mask_select(~mask)

    for  i in range(1+aug_positives):
        if distance == 'EU':
            sim_ap = torch.cdist(data_a, data_p[i*batchsize:(i+1)*batchsize,:], p=2) # batchsize * batchsize 
            distance_ap.append(torch.diag(sim_ap)) # batchsize * 1
        if distance =='Cosine':
            sim_ap = torch.cosine_similarity(data_a,data_p[i*batchsize:(i+1)*batchsize,:],dim=1)
            distance_ap.append(sim_ap)
    for i in range(n_negatives):
        if distance == 'EU':
            sim_an = torch.cdist(data_a,data_n[i*batchsize:(i+1)*batchsize,:],p=2)
            distance_an.append(torch.diag(sim_an))
        if distance == 'Cosine':
            sim_an = torch.cosine_similarity(data_a,data_n[i*batchsize:(i+1)*batchsize,:],dim=1) # batchsize * batchsize
            distance_an.append(sim_an)

    # select and combine multiple positives
    all_losses =0
    sum_pos = []
    cnt = 0
    for i in range(distance_matrix.shape[0]):
        pos = torch.nonzero(label_all[i,:])
        neg = torch.nonzero(~label_all[i,:])
        sum_neg = []
        for j in pos:
            if distance == 'Cosine':
                index_valid = torch.where(distance_matrix[i,neg[0]]>=distance_matrix[i,j])[0]
            
            #neg[torch.where(index==0)] = pos[torch.where(index ==0)] # we just use the negative samples where the flag is 1 in triplet_index
            #neg_2 = torch.where(neg_1>pos,pos,neg_1) # add constraint 
            elif distance == 'EU':
                index_valid = torch.where(distance_matrix[i,neg[0]]<=distance_matrix[i,j])[0]
            neg_valid= distance_matrix[index_valid]
            if distance == 'Cosine':
                z = neg_valid - distance_matrix[i,j]
            elif distance == 'EU':
                z = distance_matrix[i,j] - neg_valid
    
            rank =t1(z).unsqueeze(1)
            sum_neg.append(rank)
            cnt += 1 
        sum_neg = torch.cat(sum_neg,dim=1)
        sum_1 = torch.log(1+torch.sum(sum_neg,dim=1)).unsqueeze(1)
        sum_pos.append(sum_1)
    sum_pos = torch.cat(sum_pos,dim=1)
    all_losses = torch.sum(sum_pos,dim=1)
    loss = torch.sum(all_losses)/cnt 
    return loss

'''

def Ranking_loss_1(data_dict,  nclasses,M, distance, device):
    device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    anchor = []
    pos_1 = []
    pos_2 = []
    pos_3 = []
    pos_4 = []
    pos_5 = []
    neg = []
    n = M 
    for label, class_data in data_dict.items():
       # anchor_index = torch.randint(0, class_data.size(0), (1,))]
        anchor.extend(class_data)
        
        #anchor.extend(class_data[anchor_index].repeat(n,1))
        #pos.extend([item for idx, item in enumerate(class_data) if idx != anchor_index.item()])
        indices = torch.randperm(class_data.size(0))
        pos_1.extend(class_data[indices])
        indices_2 = torch.randperm(class_data.size(0))
        pos_2.extend(class_data[indices_2])
        indices_3 = torch.randperm(class_data.size(0))
        pos_3.extend(class_data[indices_3])
        indices_4 = torch.randperm(class_data.size(0))
        pos_4.extend(class_data[indices_4])
        indices_5 = torch.randperm(class_data.size(0))
        pos_5.extend(class_data[indices_5])
        neg.extend(torch.stack([value for key, value in data_dict.items() if key != label]))
        
    anchor = torch.stack(anchor).to(device)
    #pos = torch.stack(pos).to(device)
    pos_1 = torch.stack(pos_1).to(device)
    pos_2 = torch.stack(pos_2).to(device)
    pos_3 = torch.stack(pos_3).to(device)
    pos_4 = torch.stack(pos_4).to(device)
    pos_5 = torch.stack(pos_5).to(device)
    neg = torch.cat(neg, dim=0).to(device)
    neg_new = []
    #step = (nclasses -1) * M

    step = (nclasses-1) * M 
    for i in range(0, neg.size(0), step):
        group = neg[i:i + step]
        if(group.shape[0]==1):
            neg_new.extend(group.repeat(n,1))
        else:
            selected_indices = random.sample(range(group.size(0)), n )
            selected_tensors = group[selected_indices]
            neg_new.extend(selected_tensors)
    neg_new = torch.stack(neg_new).to(device)
   

    if distance == 'Cosine':
       # ap_dis = -F.cosine_similarity(anchor, pos, dim=1)
        ap_dis_1 = -F.cosine_similarity(anchor, pos_1, dim=1)
        ap_dis_2 = -F.cosine_similarity(anchor, pos_2, dim=1)
        an_dis = -F.cosine_similarity(anchor, neg_new, dim=1)
        
    elif distance == 'EU':
       # ap_dis= torch.pairwise_distance(anchor,pos,keepdim=True)
        ap_dis_1= torch.pairwise_distance(anchor,pos_1,keepdim=True)
        ap_dis_2= torch.pairwise_distance(anchor,pos_2,keepdim=True)
        ap_dis_3= torch.pairwise_distance(anchor,pos_3,keepdim=True)
        ap_dis_4= torch.pairwise_distance(anchor,pos_4,keepdim=True)
        ap_dis_5= torch.pairwise_distance(anchor,pos_5,keepdim=True)
       
        an_dis =torch.cdist(anchor,neg_new)
       
    sum_neg = []
    
    for i in range(0,nclasses):
        ap_1 = ap_dis_1[i*M:(i+1)*M]
        ap_2 = ap_dis_2[i*M:(i+1)*M]
        ap_3 = ap_dis_3[i*M:(i+1)*M]
        ap_4 = ap_dis_4[i*M:(i+1)*M]
        ap_5 = ap_dis_5[i*M:(i+1)*M]
        ap = torch.cat((ap_1,ap_2,ap_3,ap_4,ap_5),dim=0)
        ap[ap == 1] = 0
        an = an_dis[i*M:(i+1)*M].repeat(5,1)
        diff = an - ap
        diff = -diff
        mask = diff>=0
        sigmoid_result = torch.where(mask, torch.sigmoid(diff), torch.zeros_like(diff))
        sum_neg.append(sigmoid_result.sum(dim=1,keepdim=True))
    loss = torch.mean(torch.atan(torch.stack(sum_neg)))
    return loss
        
def Ranking_loss_2(data_dict,  nclasses,M, distance, device):
    device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    data = []
    for label, class_data in data_dict.items():
       # anchor_index = torch.randint(0, class_data.size(0), (1,))]
       data.extend(class_data)
        
        #anchor.extend(class_data[anchor_index].repeat(n,1))
        #pos.extend([item for idx, item in enumerate(class_data) if idx != anchor_index.item()])

    data_all = torch.stack(data).to(device)
   
    
   

    if distance == 'Cosine':
        dis_matrix = torch.cosine_similarity(data_all, data_all).to(device)
        
    elif distance == 'EU':
       # ap_dis= torch.pairwise_distance(anchor,pos,keepdim=True)
        dis_matrix =torch.cdist(data_all,data_all)
       
    ap = []
    an = []
    for i in range(0,nclasses):
        ap.extend(dis_matrix[i*M:(i+1)*M,i*M:(i+1)*M])
        t = dis_matrix[i*M:(i+1)*M].clone()
        t[:,i*M:(i+1)*M] =math.inf
        an.extend(t)
    ap = torch.stack(ap).to(device).repeat(1,nclasses)
    an = torch.stack(an).to(device)
    sum_neg =torch.zeros(ap.shape[0],M).to(device)
    for i in range(5):
        diff = an - ap
        diff = -diff
        mask = diff>=0
        sigmoid_result = torch.where(mask, torch.sigmoid(diff), torch.zeros_like(diff))
        sigmoid_result = torch.reshape(sigmoid_result,(-1,nclasses,M))
        sum_neg += sigmoid_result.sum(dim=1)
        permuted_indices = torch.randperm(diff.shape[1])
        ap = ap[:, permuted_indices]
        
    
    


#    if distance == 'Cosine':
 #       valid_indices = torch.where(distance_matrix[neg_indices[:, 0], neg_indices[:, 1]] >= distance_matrix[pos_indices[:, 0], pos_indices[:, 1]])[0]
  #  elif distance == 'EU':
   #     valid_indices = torch.where(distance_matrix[neg_indices[:, 0], neg_indices[:, 1]] <= distance_matrix[pos_indices[:, 0], pos_indices[:, 1]])[0]

    loss = torch.mean(torch.atan(sum_neg))
    return loss

def Ranking_loss_3(anchor_data,pos_data,neg_data,distance,device):
    if distance == 'Cosine':
          
        ap_dis = -F.cosine_similarity(anchor_data, pos_data, dim=1)
        an_dis = -F.cosine_similarity(anchor_data, neg_data, dim=1)
    elif distance == 'EU':
        
        ap_dis= torch.pairwise_distance(anchor_data,pos_data,keepdim=True)
        an_dis =torch.cdist(anchor_data,neg_data)
    
    diff = an_dis - ap_dis
    diff = -diff
    mask = diff>=0
    sum_neg = []
    sigmoid_result = torch.where(mask, torch.sigmoid(diff), torch.zeros_like(diff))
    sum_neg.append(sigmoid_result.sum(dim=1,keepdim=True))
    loss = torch.mean(torch.atan(torch.stack(sum_neg)))
    return loss

def Ranking_loss_kmeans(anchor,pos,neg,distance,aug_number,batchsize,number,device):
    device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    ap_dis = {}
    diff = {}
    aug_number =aug_number+1
    if distance == 'Cosine':
        for i in range(aug_number):
            ap_dis[i] = -F.cosine_similarity(anchor,pos[i],dim=1)
            
        an_dis = -F.cosine_similarity(anchor,neg,dim=1)
    elif distance == 'EU':
        for i in range(aug_number):
            ap_dis[i] = torch.pairwise_distance(anchor,pos[i],keepdim=True).to(device)
       
        an_dis = torch.pairwise_distance(anchor,neg,keepdim=True).to(device)
    for i in range(aug_number): 
        diff[i]=an_dis - ap_dis[i].T
    loss = 0 
    for i in range(aug_number):
        for j in range(batchsize):
                diff_1 = diff[i].T
                diff_2 = -diff_1[number*j:number*(j+1),number*j:number*(j+1)]
                mask = diff_2>=0
                sum_neg = []
                sigmoid_result = torch.where(mask, torch.sigmoid(diff_2), torch.zeros_like(diff_2))
                sum_neg.append(sigmoid_result.sum(dim=1,keepdim=True))
                loss += torch.mean(torch.atan(torch.stack(sum_neg)))
    if(aug_number!=0):
        loss = loss/(aug_number*batchsize)
    return loss.to(device)
    