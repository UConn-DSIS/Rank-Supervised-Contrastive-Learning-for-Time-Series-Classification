import torch
from torch import nn
import torch.nn.functional as F
from loss.Ranking_loss import Ranking_loss
from utils.utils import triplet_index
def hierarchical_rank_loss(data_a, data_p,data_n,n_negatives,batchsize,number,distance,aug_positives,alpha=0.5, temporal_unit=0,):
    # normalize the emebeddings 
    data_a = F.normalize(data_a,dim=2)  
    data_p = F.normalize(data_p,dim=2)
    data_n = F.normalize(data_n,dim=2)
    # calculate the triplet_index
    triplet_matrix = triplet_index(data_a,data_p,data_n,n_negatives,aug_positives) 
    loss = torch.tensor(0., device=data_a.device)
    d = 0
    while data_a.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_rank_loss(data_a, data_p,data_n,n_negatives,batchsize,number,distance,aug_positives,triplet_matrix)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(data_a,data_p[0:batchsize,:])
        d += 1
        data_a = F.max_pool1d(data_a.transpose(1,2), kernel_size=2).transpose(1,2)
        data_p = F.max_pool1d(data_p.transpose(1,2), kernel_size=2).transpose(1,2)
        data_n = F.max_pool1d(data_n.transpose(1,2), kernel_size=2).transpose(1,2)
    if data_a.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_rank_loss(data_a,data_p,data_n,n_negatives,batchsize,number,distance,aug_positives,triplet_matrix)
        d += 1
    return loss / d

def instance_rank_loss(data_a, data_p,data_n,n_negatives,batchsize,number,distance,aug_positives,triplet_matrix):
    n_negatives = n_negatives + number
   
    # calculate the sim
    distance_ap = []
    distance_an = []
    t1 = torch.nn.Sigmoid()
    data_a = data_a.reshape((data_a.shape[0],-1))
    data_p = data_p.reshape((data_p.shape[0],-1))
    data_n = data_n.reshape((data_n.shape[0],-1))
    for  i in range(1+aug_positives):
        if distance == 'EU':
            sim_ap = torch.cdist(data_a, data_p[i*batchsize:(i+1)*batchsize,:], p=2) # batchsize * batchsize 
            distance_ap.append(torch.diag(sim_ap)) # batchsize * 1
        if distance =='Cosine':
            sim_ap = torch.cosine_similarity(data_a,data_p[i*batchsize:(i+1)*batchsize,:],dim=1)
            distance_ap.append(sim_ap)
    for i in range(n_negatives):
        if distance == 'EU':
            sim_an = torch.cdist(data_a.reshape(data_a.shape[0],-1),data_n[i*batchsize:(i+1)*batchsize,:].reshape(data_a.shape[0],-1),p=2)
            distance_an.append(torch.diag(sim_an))
        if distance == 'Cosine':
            sim_an = torch.cosine_similarity(data_a,data_n[i*batchsize:(i+1)*batchsize,:],dim=1) # batchsize * batchsize
            distance_an.append(sim_an)
    # select and combine multiple positives
    all_losses =0
    sum_pos = []
    for i in range(1+aug_positives):
        pos = distance_ap[i]
        sum_neg = []
        for j in range(n_negatives):
            neg = distance_an[j]
            index = triplet_matrix[i][j]
            neg_1= torch.where(index==0,pos,neg)
            #neg[torch.where(index==0)] = pos[torch.where(index ==0)] # we just use the negative samples where the flag is 1 in triplet_index
            #neg_2 = torch.where(neg_1>pos,pos,neg_1) # add constraint 
            if distance == 'Cosine':
                z = neg_1 - pos
            elif distance == 'EU':
                z = pos - neg_1
        #print(z)
            rank =t1(z).unsqueeze(1)
            sum_neg.append(rank)
        
        sum_neg = torch.cat(sum_neg,dim=1)
        sum_1 = torch.log(1+torch.sum(sum_neg,dim=1)).unsqueeze(1)
        sum_pos.append(sum_1)
        #rank = torch.max(z, torch.tensor([0.]).to(device))
        #print('rank',rank)
        #sum = torch.sum(rank,dim=1)
        #print('sum',sum)
    #
    sum_pos = torch.cat(sum_pos,dim=1)
    all_losses = torch.sum(sum_pos,dim=1)
    loss = torch.sum(all_losses)/((1+aug_positives)*batchsize) 
    return loss

def temporal_contrastive_loss(data_a,data_p):
    B,T = data_a.size(0),data_a.size(1)
    if T == 1:
        return data_a.new_tensor(0.)
    z = torch.cat([data_a, data_p], dim=1)  # B x 2T 
    sim = torch.matmul(z, z.transpose(1,2))  # 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:,:, :-1]    # 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:,:, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=data_a.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
