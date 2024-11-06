''' 
Try to change the encoder architechture and add fine_tune 
''' 
import os
import torch
import argparse
from utils.load_data import load_UCR_dataset,load_UEA
from utils.utils import setup_seed,plot_TSNE,define_model,_logger
import torch.utils.data as Data
from torch.utils.data import DataLoader, Sampler
from sklearn.model_selection import train_test_split
import logging
import torch.nn  
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
from utils.train import train_encoder,train_model
from utils.classification import classification
from utils.ts2vec import TS2Vec
from utils._eval_protocols import eval_classification
import torch.nn.functional as F
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    # parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
    #                     help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--device', type=int, default=0,metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    # parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
    #                     help='path of the file of hyperparameters to use; ' +
    #                          'for training; must be a JSON file')
    # parser.add_argument('--load', action='store_true', default=False,
    #                     help='activate to load the estimator instead of ' +
    #                          'training it')
    parser.add_argument('--model', metavar='ARCH', default='resnet18',
                    help='model architecture: '+ ' (default: resnet50)')
    parser.add_argument('--batchsize',type=int,default=128,required=True,help='batchsize')
    #parser.add_argument('--n_negatives',type=int,default=50,required=True,help='number of batchs of negative samples')
    #parser.add_argument('--number',type=int,default=5,required=True,help = 'number of mix up negative samples')
    parser.add_argument('--distance', default='EU', type=str,help='chose of the similarity measurement')
    parser.add_argument('--aug_positives',type=int,default=5,required = True,help = 'number of augmentaed positive samples')
    parser.add_argument('--seed',type=int,default=128,required=True,help="set up the seed")
    parser.add_argument('--lr',type=float,default=0.00001,required=True,help='learning rate of encoder')
    parser.add_argument('--weight_decay',type=float,default=0.0005,required=True,help='wieght decay of the encoder')
    parser.add_argument('--epochs_up',type=int,default=300,required=True,help='epochs for the encdoer')
    parser.add_argument('--epochs_down',type=int, default=100,required=True,help='epochs for the downstream')    
    parser.add_argument('--loss',type=str,default='rank',required=True,help='choose the loss function of hierarchical architecture')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    # set up the seed, make sure the reproducibility
    setup_seed(args.seed)   
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False
    if args.path == 'UCR':
        logs_save_dir = 'UCR_results'    
        train_all, train_labels, test, test_labels,nclasses= load_UCR_dataset(
            args.path, args.dataset
        )
    elif args.path == 'UEA':
        logs_save_dir = 'UEA_results'
        train_all,train_labels,test,test_labels,nclasses = load_UEA(args.dataset)
    
    train_all[np.isnan(train_all)]=0
    test[np.isnan(test)]=0
    experiment_log_dir = os.path.join('/home/qir21001/AAAI2022/',logs_save_dir, args.dataset)
    os.makedirs(experiment_log_dir, exist_ok=True)
    # logging 
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.dataset}')
    logger.debug("=" * 45)
    logger.debug("Data loaded----------")
    logger.debug(f"train_shape_before{train_all.shape}")
    logger.debug(f"Batchsize: {args.batchsize}  Epochs_up:{args.epochs_up} LR:{args.lr} Aug_positives:{args.aug_positives} Backbone Model:{args.model} Loss:{args.loss}")
    logger.debug(f"Seed:{args.seed}")
   
    train_all = np.transpose(train_all,axes=(0,2,1))
    test = np.transpose(test,axes=(0,2,1))
    logger.debug(f"train_shape_after{train_all.shape}")
    #plot_TSNE(train_all,train_labels,file_path='/home/qir21001/AAAI2022/TSNE/'+args.dataset+'_Raw_data',nclasses=nclasses)
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    
    
    train_all = torch.from_numpy(train_all).to(torch.float)
    train_labels = torch.from_numpy(train_labels).to(torch.float)
    test = torch.from_numpy(test).to(torch.float)
    test_labels = torch.from_numpy(test_labels).to(torch.float)
    
    print("Labels Shape",train_labels.shape)
    # split the train_dataset
   # train_set,val_set,train_label,val_label = train_test_split(train_all,train_labels,test_size=0.2,random_state=46,stratify=train_labels)
    train_dataset= torch.utils.data.TensorDataset(train_all,train_labels)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print("Number of classes ",nclasses)
   # generate the test dataset, val-dataset
    test = test.to(device)
    test_labels = test_labels.to(device)
    #val_set = val_set.to(device)
    #val_label = val_label.to(device)
    # According to the model to train the dataset
    # plot the TSNE before training:
    #plot_TSNE(train_all,train_labels,file_path='/home/qir21001/AAAI2022/TSNE/Ablation/'+args.dataset+'_'+str(args.batchsize)+'_'+args.model+str(args.aug_positives)+'_Raw_data',nclasses=nclasses)

        
    train_encoder(args.dataset,train_loader,args.model,args.device,logger=logger,train=train_all,train_labels=train_labels,nclasses=nclasses,loss=args.loss,seed=args.seed,lr=args.lr,aug_positives=args.aug_positives,weight_decay=args.weight_decay,epochs_up=args.epochs_up,batchsize=args.batchsize,distance=args.distance)
    print("traing_finished: save model")
    save_dir = '/home/qir21001/AAAI2022/result_model/'+args.dataset+'_'+args.model+'_'+str(args.batchsize)+'_'+str(args.aug_positives)+'_'+str(args.lr)+'_'+str(args.epochs_up)+'_'+str(args.seed)
    state_dict = torch.load(os.path.join(save_dir,'Rank_model.pt'))    
    model = define_model(args.model,input_dim = train_all.shape[1])
 
        # for k in list(state_dict.keys()):                     
        #     if k.startswith('backbone.'):
        #         if k.startswith('backbone') and not k.startswith('backbone.fc'):
        #     # remove prefix
        #             state_dict[k[len("backbone."):]] = state_dict[k]
        #     del state_dict[k]
    log = model.load_state_dict(state_dict, strict=False)
    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False
    #parameters = list(filter(lambda p: p.requires_grad, model.parameters())) #
    model = model.to(device)
    # plot the TSNE after the encoder training:
    h = train_model(train_all,args.model,model,device,encode=True)
    h = F.normalize(h,dim=1)
    #plot_TSNE(h,train_labels,file_path='/home/qir21001/AAAI2022/TSNE/Ablation/'+args.dataset+'_'+str(args.batchsize)+'_'+args.model+str(args.aug_positives)+'_Encoder_repre',nclasses=nclasses)
    #train_label = train_label.type(torch.LongTensor)
    #train_dataset = torch.utils.data.TensorDataset(train_set,train_label)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,num_workers=12, pin_memory=True, drop_last=True)
    # if hasattr(torch.cuda, 'empty_cache'):
    # 	torch.cuda.empty_cache()
    #classification(args.dataset,nclasses,train_loader,train_all,train_labels,val_set,val_label,test,test_labels,args.device,model,args.model,classifier,logger,epochs=args.epochs_down,lr_model=1e-4,lr_classifier=1e-3,weight_decay=0.0008)
    out, eval_res = eval_classification(model, args.model, train_all, train_labels, test, test_labels, args.dataset,logger,args.device,eval_protocol='svm')