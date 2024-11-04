import os
import torch
import numpy as np
import pickle
import random
import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/qir21001/TNC")
from data.load_data import load_UCR_dataset,load_UEA
from tnc.models import RnnEncoder, StateClassifier, E2EStateClassifier, WFEncoder, WFClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score,precision_score,f1_score
from sklearn.metrics import average_precision_score
from datetime import datetime
import logging
import sys
from tnc.utils import plot_TSNE
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_run(model, dataloader, train=False, lr=0.01):
    if train:
        model.train()
    else:
        model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    for x, y in dataloader:
        y = y.to(device)
        x = x.to(device)
        prediction = model(x)
        state_prediction = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, y.long())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_all.append(y.cpu().detach().numpy())
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())

        epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
        epoch_loss += loss.item()
        batch_count += 1
    del x, y
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc, c


def run_test_down(encoder, classifier, x,y, lr=0.01):
    
    classifier.eval()
    encoder.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

    
   
    y = y.to(device)
    x = x.to(device)
    encodings = encoder(x)
    prediction = classifier(encodings)
    state_prediction = torch.argmax(prediction, dim=1)
    loss = loss_fn(prediction, y.long())
    acc = torch.eq(state_prediction, y).sum().item()/len(x)
    
  
   
    prediction = prediction.cpu().detach().numpy()
    prediction_class_all = np.argmax(prediction, -1)
    y = y.cpu().detach().numpy()
    y_onehot_all = np.zeros(prediction.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y.astype(int)] = 1
   # epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    #epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    #c = confusion_matrix(y_all.astype(int), prediction_class_all)
    f1 = f1_score(y.astype(int),prediction_class_all,average='macro')
    precision = precision_score(y.astype(int),prediction_class_all,average='macro')
    return acc,f1,precision

def epoch_run_encoder(encoder, classifier, dataloader, train=False, lr=0.01):
    if train:
        classifier.train()
        encoder.train()
    else:
        classifier.eval()
        encoder.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    for x, y in dataloader:
        y = y.to(device)
        x = x.to(device)
        
        encodings = encoder(x)
        prediction = classifier(encodings)
        if prediction.ndim==1:
            prediction = prediction.view(1,-1)
        state_prediction = torch.argmax(prediction, dim=-1)
        #print(prediction)
        #print("prediction_shape",prediction.shape)
        #print("y_shape",y.shape)
        loss = loss_fn(prediction, y.long())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_all.append(y.cpu().detach().numpy())
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())

        epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
        epoch_loss += loss.item()
        batch_count += 1
    del x, y
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
   # epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    #epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    #c = confusion_matrix(y_all.astype(int), prediction_class_all)
    f1 = f1_score(y_all.astype(int),prediction_class_all,average='macro')
    precision = precision_score(y_all.astype(int),prediction_class_all,average='macro')
    return epoch_loss / batch_count, epoch_acc / batch_count,f1,precision


def train(train_loader, valid_loader, classifier, lr, data_type, encoder=None, n_epochs=100, type='e2e', cv=0):
    best_auc, best_acc, best_aupc, best_loss = 0, 0, 0, np.inf
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    for epoch in range(n_epochs):
        if type=='e2e':
            train_loss, train_acc, train_auc, train_auprc, _ = epoch_run(classifier, dataloader=train_loader, train=True, lr=lr)
            test_loss, test_acc, test_auc, test_auprc,  _ = epoch_run(classifier, dataloader=valid_loader, train=False)
        else:
            train_loss, train_acc, f1,prec = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=train_loader, train=True, lr=lr)
            test_loss, test_acc, f1,prec  = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=valid_loader, train=False)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_loss<best_loss:
            
            best_acc = test_acc
            best_loss = test_loss
           
            if type == 'e2e':
                state = {
                    'epoch': epoch,
                    'state_dict': classifier.state_dict(),
                    'best_accuracy': test_acc,
                    'best_accuracy': best_auc
                }
            else:
                state = {
                    'epoch': epoch,
                    'state_dict': torch.nn.Sequential(encoder, classifier).state_dict(),
                    'best_accuracy': test_acc,
                    'best_accuracy': best_auc
                }
            if not os.path.exists( './ckpt/classifier_test/%s'%data_type):
                os.mkdir( './ckpt/classifier_test/%s'%data_type)
            torch.save(state, './ckpt/classifier_test/%s/%s_checkpoint_%d.pth.tar'%(data_type, type, cv))

    # Save performance plots
    '''
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses, label="train Loss")
    plt.plot(np.arange(n_epochs), test_losses, label="test Loss")

    plt.plot(np.arange(n_epochs), train_accs, label="train Acc")
    plt.plot(np.arange(n_epochs), test_accs, label="test Acc")
    plt.savefig(os.path.join("./plots/%s" % data_type, "classification_%s_%d.pdf"%(type, cv)))
    '''
    return best_acc, best_auc, best_aupc


def run_test(dataset, e2e_lr, tnc_lr, cpc_lr, trip_lr, data_path, window_size, n_cross_val,logger):
    # Load data
    if(data_path =='UCR'):
        path = '/home/qir21001/AAAI2022/UCR'
        X_train, y_train,X_test,y_test,nclasses = load_UCR_dataset(path,dataset)
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)]=0
    elif(data_path == 'UEA'):
        X_train, y_train,X_test,y_test,nclasses = load_UEA(dataset)
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)]=0
        # change the shape of the dataset
    X_train = np.transpose(X_train,(0,2,1))
    logger.debug(f"Changed X_train shape:{X_train.shape}")
    X_test = np.transpose(X_test,(0,2,1))
    logger.debug(X_test.shape)
    logger.debug(y_test.shape)
    T = X_train.shape[-1]
    dim = X_train.shape[1]
    X_train = torch.from_numpy(X_train).to(torch.float)
    y_train = torch.from_numpy(y_train).to(torch.float)
    X_test = torch.from_numpy(X_test).to(torch.float)
    y_test = torch.from_numpy(y_test).to(torch.float)
   

    
    
    tnc_accs, tnc_aucs, tnc_auprcs,tnc_f1,tnc_pre = [], [], [],[],[]
   
    for cv in range(n_cross_val):
        shuffled_inds = list(range(len(X_train)))
        random.shuffle(shuffled_inds)
        X_train = X_train[shuffled_inds]
        y_train = y_train[shuffled_inds]
        n_train = int(0.8*len(X_train))
        x_train, x_test = X_train[:n_train], X_train[n_train:]
        y_train_1, y_test_1 = y_train[:n_train], y_train[n_train:]

        trainset = torch.utils.data.TensorDataset(x_train, y_train_1)
        validset = torch.utils.data.TensorDataset(x_test, y_test_1)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=8, shuffle=False)

       
          

       
        encoding_size = 320
        tnc_encoder = RnnEncoder(hidden_size=10,in_channel=dim, encoding_size=encoding_size, device=device)
        save_path = os.path.join('/home/qir21001/TNC/ckpt',dataset,'encoder_checkpoint')

    
        tnc_checkpoint = torch.load('%s/checkpoint_pth.tar' % (save_path))
        print(tnc_checkpoint.keys())
        tnc_encoder.load_state_dict(tnc_checkpoint['encoder_state_dict'])
        X_train_repr = tnc_encoder(torch.Tensor(X_train))
        file_path = '/home/qir21001/TNC/TSNE/'+'TNC_'+args.dataset+'Encoder_repr' 
     #   plot_TSNE(X_train_repr,y_train,file_path=file_path)
        tnc_classifier = WFClassifier(encoding_size=320, output_size=nclasses).to(device)
        tnc_model = torch.nn.Sequential(tnc_encoder, tnc_classifier).to(device)
        n_epochs = 100

        

        # Train the model

        # ***** TNC *****
        best_acc_tnc, best_auc_tnc, best_auprc_tnc = train(train_loader, valid_loader, tnc_classifier, tnc_lr,
                                           encoder=tnc_encoder, data_type=dataset, n_epochs=n_epochs, type='tnc', cv=cv)
       
    
    
    test_acc_tnc,test_f1,test_precision = run_test_down(tnc_encoder, tnc_classifier, X_test, y_test)
    logger.debug('=======> Performance Summary:')
    logger.debug('TNC model: \t Accuracy: %.3f \t  F1: %.3f \t Precision: %.3f'% (test_acc_tnc, test_f1,test_precision))
   

'''
        with open("./outputs/%s_classifiers.txt"%dataset, "a") as f:
            f.write("\n\nPerformance result for a fold" )
            
            f.write("TNC model: \t AUC: %s\t Accuracy: %s \n \n" % (str(best_auc_tnc), str(best_acc_tnc)))
           '''

    
    


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Run classification test')
    parser.add_argument('--dataset', type=str, default='simulation')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--w', type=float, default=0.05)
    parser.add_argument('--path', type=str,default='UCR')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--batchsize',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    if not os.path.exists('./ckpt/classifier_test'):
        os.mkdir('./ckpt/classifier_test')
    logs_save_dir = 'experiments_logs'
    experiment_description='Exp1'
    run_description='test'
    experiment_log_dir = os.path.join('/home/qir21001/TNC',logs_save_dir, experiment_description, run_description + f"_seed_{args.seed}")
    os.makedirs(experiment_log_dir, exist_ok=True)
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.dataset}')
    
    logger.debug(f'Mode:    test')
    logger.debug("=" * 45)
    logger.debug(f"Batchszie{args.batchsize}")
    
    if args.path == 'UCR':
        path = os.path.join('/home/qir21001/AAAI2022/UCR',args.dataset)
        run_test(dataset=args.dataset,e2e_lr=0.0001, tnc_lr=0.0001, cpc_lr=0.01, trip_lr=0.01,
                 data_path=args.path, window_size=5, n_cross_val=args.cv,logger=logger)
    elif args.path == 'UEA':
        path = os.path.join('/home/qir21001/AAAI2022/UEA',args.dataset)
        run_test(dataset=args.dataset,e2e_lr=0.0001, tnc_lr=0.0001, cpc_lr=0.01, trip_lr=0.01,
                 data_path=args.path, window_size=5, n_cross_val=args.cv,logger=logger)
