"""
Temporal Neighborhood Coding (TNC) for unsupervised learning representation of non-stationary time series
"""

import torch
import matplotlib.pyplot as plt
import argparse
import math
import seaborn as sns; sns.set()
import sys,os
#BASE_DIR = os.path.dirname(os.path.dirname(os.abspath(__file__)))
#sys.path.append(BASE_DIR)
from torch.utils import data
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from tnc.models import RnnEncoder, WFEncoder
from tnc.utils import plot_distribution, track_encoding
from tnc.evaluations import WFClassificationExperiment, ClassificationPerformanceExperiment
from statsmodels.tsa.stattools import adfuller
from data.load_data import load_UCR_dataset,load_UEA
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size

        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))


class TNCDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, window_size, augmentation, epsilon=3, state=None, adf=False):
        super(TNCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.window_size = window_size
        self.sliding_gap = int(window_size*25.2)
        self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]
    
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)

        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size,4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n


def epoch_run(loader, disc_model, encoder, device, w=0, optimizer=None, train=True):
    if train:
        encoder.train()
        disc_model.train()
    else:
        encoder.eval()
        disc_model.eval()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    encoder.to(device)
    disc_model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    for x_t, x_p, x_n, _ in loader:
        mc_sample = x_p.shape[1]
        batch_size, f_size, len_size = x_t.shape
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = np.repeat(x_t, mc_sample, axis=0)
        neighbors = torch.ones((len(x_p))).to(device)
        non_neighbors = torch.zeros((len(x_n))).to(device)
        x_t, x_p, x_n = x_t.to(device), x_p.to(device), x_n.to(device)

        z_t = encoder(x_t)
        z_p = encoder(x_p)
        z_n = encoder(x_n)

        d_p = disc_model(z_t, z_p)
        d_n = disc_model(z_t, z_n)

        p_loss = loss_fn(d_p, neighbors)
        n_loss = loss_fn(d_n, non_neighbors)
        n_loss_u = loss_fn(d_n, neighbors)
        loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p_acc = torch.sum(torch.nn.Sigmoid()(d_p) > 0.5).item() / len(z_p)
        n_acc = torch.sum(torch.nn.Sigmoid()(d_n) < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc+n_acc)/2
        epoch_loss += loss.item()
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count


def learn_encoder(x, encoder, dataset,window_size,batch_size, w, lr=0.0001, decay=0.005, mc_sample_size=20,
                  n_epochs=10, path='simulation', device='cpu', augmentation=1, n_cross_val=1, cont=False):
    accuracies, losses = [], []
    for cv in range(n_cross_val):
        
        if 'UCR' in path:
            encoder = RnnEncoder(hidden_size=100, in_channel=x.shape[1],encoding_size=320, device=device)
            
        elif 'UEA' in path:
            encoder = RnnEncoder(hidden_size=10, in_channel=x.shape[1], encoding_size=320, device=device)
            
        save_path = os.path.join('/home/qir21001/TNC/ckpt',dataset,'encoder_checkpoint')
       
        os.makedirs(save_path,exist_ok=True)
        if cont:
            checkpoint = torch.load('%s/checkpoint_%d.pth.tar'%(save_path, cv))
            encoder.load_state_dict(checkpoint['encoder_state_dict'])

        disc_model = Discriminator(encoder.encoding_size, device)
        params = list(disc_model.parameters()) + list(encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        performance = []
        best_acc = 0
        best_loss = np.inf

        for epoch in range(n_epochs+1):
            trainset = TNCDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size,
                                  window_size=window_size, augmentation=augmentation, adf=True)
            train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
            validset = TNCDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=mc_sample_size,
                                  window_size=window_size, augmentation=augmentation, adf=True)
            valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)

            epoch_loss, epoch_acc = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer,
                                              w=w, train=True, device=device)
            test_loss, test_acc = epoch_run(valid_loader, disc_model, encoder, train=False, w=w, device=device)
            performance.append((epoch_loss, test_loss, epoch_acc, test_acc))
            if epoch%10 == 0:
                print('(cv:%s)Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
                      % (cv, epoch, epoch_loss, epoch_acc, test_loss, test_acc))
            if best_loss > test_loss or path=='har':
                best_acc = test_acc
                best_loss = test_loss
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'discriminator_state_dict': disc_model.state_dict(),
                    'best_accuracy': test_acc
                }
                torch.save(state, '%s/checkpoint_pth.tar'%(save_path))
        accuracies.append(best_acc)
        losses.append(best_loss)
        # Save performance plots
        '''
        if not os.path.exists('./plots/%s'%path):
            os.mkdir('./plots/%s'%path)
        train_loss = [t[0] for t in performance]
        test_loss = [t[1] for t in performance]
        train_acc = [t[2] for t in performance]
        test_acc = [t[3] for t in performance]
        plt.figure()
        plt.plot(np.arange(n_epochs+1), train_loss, label="Train")
        plt.plot(np.arange(n_epochs+1), test_loss, label="Test")
        plt.title("Loss")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%path, "loss_%d.pdf"%cv))
        plt.figure()
        plt.plot(np.arange(n_epochs+1), train_acc, label="Train")
        plt.plot(np.arange(n_epochs+1), test_acc, label="Test")
        plt.title("Accuracy")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%path, "accuracy_%d.pdf"%cv))
'''
    
    print('=======> Performance Summary:')
    print('Accuracy: %.3f '%(100*np.mean(accuracies)))
    print('Loss: %.4f +- %.4f'%(np.mean(losses), np.std(losses)))
    return encoder


def main(dataset, cv, w, cont,path,batchsize):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

   
    window_size =3
    if(path =='UCR'):
        path = '/home/qir21001/AAAI2022/UCR'
        X_train, y_train,X_test,y_test,nclasses = load_UCR_dataset(path,dataset)
      
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)]=0
    elif(path == 'UEA'):
        path = '/home/qir21001/AAAI2022/UEA'
        X_train, y_train,X_test,y_test,nclasses = load_UEA(dataset)
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)]=0
        # change the shape of the dataset
    X_train = np.transpose(X_train,(0,2,1))
    print("Changed X_train shape:",X_train.shape)
    X_test = np.transpose(X_test,(0,2,1))
    encoder = RnnEncoder(hidden_size=100, in_channel=X_train.shape[1],encoding_size=320, device=device)
    learn_encoder(torch.Tensor(X_train), encoder, dataset=dataset,w=w, batch_size=batchsize,lr=1e-4, decay=1e-5, window_size=window_size, n_epochs=150,
                          mc_sample_size=20, path=path, device=device, augmentation=5, n_cross_val=cv)
   
        
    save_path = os.path.join('/home/qir21001/TNC/ckpt',dataset,'encoder_checkpoint')
    print(save_path)
    
   

    

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run TNC')
    parser.add_argument('--dataset', type=str, default='simulation')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--w', type=float, default=0.05)
    parser.add_argument('--path', type=str,default='UCR')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--batchsize',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    print("devices",device)
    print('TNC model with w=%f'%args.w)
    main(args.dataset, args.cv, args.w, args.cont,args.path,args.batchsize)


