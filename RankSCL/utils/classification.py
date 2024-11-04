import torch.nn as nn 
import torch
import torch.nn.functional as F
from utils.utils import plot_TSNE,define_model
from utils.train import train_model
import copy
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import nni
def classification(dataset,nclasses,train_loader,train_all,train_labels,val_set,val_label,test,test_labels,device,model,model_name,classifier,logger,epochs,lr_model = 1e-4,lr_classifier=1e-3,weight_decay=0.0008):
    device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':lr_model},{'params':classifier.parameters()}], lr=lr_classifier, weight_decay=0.0008)
    best_val_acc = 0
    for epoch in range(epochs):   
        for step,data in enumerate(train_loader,start=0):
            train,train_label = data
            train,train_label = train.to(device),train_label.to(device).type(torch.LongTensor)
            # print("train.dtype,train_labels.dtype:",train.dtype,train_labels.dtype)# train = train.type(torch.FloatTensor)
            # # train_labels = train_labels.type(torch.FloatTensor)
            # print("train_shape,train_labels_shape:",train.size(),train_labels.size()
            model.train()
            classifier.train()
            optimizer.zero_grad()
            h = train_model(train,model_name,model,device,encode=False)
            if(len(h.shape)>2):
                h = h.reshape(h.shape[0],-1)
            logits = classifier(h)
            # print(prediction.shape)
        
            y_pred = torch.argmax(logits,1)
            loss = criterion(logits,train_label.to(device))
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
        
            print("Epoch %d ----- loss %.3f"%(epoch+1,loss))
            train_accuracy = torch.eq(y_pred.to(device),train_label.to(device)).sum().float().item()/len(train_label)
            print("Epoch %d ---- train_acc %.3f"%(epoch+1,train_accuracy))  
        
        #validation
        model.eval()
        classifier.eval()
        h = train_model(val_set,model_name,model,device,encode=False)
        if(len(h.shape)>2):
            h = h.reshape(h.shape[0],-1)
        y_pred = torch.argmax(classifier(h),1)   
        val_accuracy = torch.eq(y_pred.to(device),val_label.to(device)).sum().float().item()/len(val_label)
        nni.report_intermediate_result(val_accuracy)
        print("Epoch %d ---- val %.3f",epoch+1,val_accuracy)
        # save the best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            print("save model")
            torch.save(model.state_dict(),'/home/qir21001/AAAI2022/result_model/'+model_name+'_'+dataset+'_downstream_classification.pt')
            
            best_model = copy.deepcopy(classifier)
            logger.info("Epoch %d ----- train_loss %.3f,train_acc %.3f, val_acc %.3f"%(epoch+1,loss,train_accuracy,val_accuracy))

    #retrain: 
    model = define_model(model_name)
    model.load_state_dict(torch.load('/home/qir21001/AAAI2022/result_model/'+model_name+'_'+dataset+'_downstream_classification.pt'))
    train_all = train_all.to(device)
    train_labels = train_labels.to(device).type(torch.LongTensor)
    model.train()
    best_model.train()
    optimizer.zero_grad()
    h = train_model(train_all,model_name,model,device,encode=False)
    if(len(h.shape)>2):
        h = h.reshape(h.shape[0],-1)
    plot_TSNE(h,train_labels,file_path='TSNE/'+dataset+'_downstream_classfication',nclasses=nclasses)    
    prediction = best_model(h)
    loss = criterion(prediction,train_labels.to(device))
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    # best_model.eval()
    # if(args.model == 'CNN'):
    #     y_pred = torch.argmax(best_model(train.unsqueeze(1).float()),1)
    # elif(args.model == 'LSTM'):
    #     y_pred = torch.argmax(best_model(train.unsqueeze(2).float()),1)
    # elif(args.model == 'dilation_CNN'):
    #     y_pred = torch.argmax(best_model(train.unsqueeze(1).float()),1)
    # elif(args.model == 'resnet'):
    #     y_pred = torch.argmax(best_model(train_1.unsqueeze(2).float()),1)
    # train_accuracy = torch.eq(y_pred.to(device),train_labels.to(device)).sum().float().item()/len(train_labels)
    # print("Final train_acc %.3f"%(train_accuracy))
    # logger.info("Final train_acc %.3f"%(train_accuracy)
    #testting
    model.eval()
    best_model.eval() 
    h = train_model(test,model_name,model,device,encode=False)
    if(len(h.shape)>2):
        h = h.reshape(h.shape[0],-1)
    # print("score",score)
    y_pred = best_model(h)
    y_pred = torch.argmax(y_pred,1)
    accuracy = torch.eq(y_pred.to(device),test_labels.to(device)).sum().float().item()/len(test_labels) 
    nni.report_final_result(accuracy)
    precision = precision_score(test_labels.cpu().numpy(),y_pred.cpu().numpy(),average="macro")
    F1 = f1_score(test_labels.cpu().numpy(),y_pred.cpu().numpy(),average="macro")

    print('Test ACC,precision_score, f1_score',accuracy,precision,f1_score)
    logger.info('Test ACC %.3f,precision_score %.3f, f1_score%.3f',accuracy,precision,F1)
    #print(y_pred)
    
    C2 = confusion_matrix(test_labels.cpu().numpy(),y_pred.cpu().numpy())
    print(C2)
    logger.info(C2)
    # normalize 
    c2=C2 / C2.astype(numpy.float).sum(axis=1)
    fig,ax = plt.subplots(figsize=(10,10))
    indices = range(len(C2))
    classes = range(nclasses)
    sns.heatmap(c2,annot=True,fmt='.2f',xticklabels=indices,yticklabels=classes,)

    
    # plt.xticks(num_local,nclasses,rotation=90)
    # plt.yticks(num_local,nclasses)
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.show(block=False)
    fig.savefig('Results/'+"Rank_theta_"+model_name+'_'+dataset+'_recall.jpg')
    plt.close()    


