import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.tsencoder import TSEncoder
from loss.hierarchical_rank_loss import hierarchical_rank_loss
from loss.hierarchical_contrastive_loss import hierarchical_contrastive_loss
import math
from utils.utils import aug_data,generate_negatives
import random
class generate_data():
    def __init__(self,data,labels,n_negatives,n_triplets):
        self.n_negatives = n_negatives
        self.data =data
        self.labels = labels
        self.n_triplets = n_triplets
        self.train_triplets = self.generate_triplets(self.labels,self.n_negatives)
        
    def generate_triplets(self,labels,n_negatives):
        triplets = []
        for i in range(self.n_triplets):
            idx = np.random.randint(0,labels.shape[0])
            idx_matches = list(np.where(labels==labels[idx])[0])
            idx_ap = random.sample(idx_matches,2)
            idx_a,idx_p = idx_ap[0],idx_ap[1]
            n_list   = []
            for x in range(self.n_negatives):
                idx_no_matches = list(np.where(labels!=labels[idx])[0])
                idx_n = random.sample(idx_no_matches,1)[0]
                n_list.append(idx_n)
            triplets.append([idx_a,idx_p]+n_list)
        return np.array(triplets)
    def __getitem__(self,index):
        t = self.train_triplets[index]
        a,p,n = self.data[t[0]],self.data[t[1]],self.data[t[2:]]
        return a,p,n
        
    def __len__(self):
        return self.train_triplets.shape[0]
#  reshape the negative samples: from [n_negatives * batchsize, time_step] to [batchsize * negatives,time_step]

class TS2Vec:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320, 
        hidden_dims=64, 
        depth=10, 
        device=0, 
        lr=0.001, 
        batch_size=16,
        n_negatives=0,
        aug_positives=0,
        temporal_unit=0, 
        number = 0,
        distance = 'EU'
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.batch_size = batch_size
        #self.max_train_length = max_train_length
        self.n_negatives = n_negatives
        self.aug_positives = aug_positives
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.n_epochs = 0
        self.n_iters = 0
        self.number =number
        self.distance = distance
    
    def fit(self, train_data, train_labels,logger,loss,n_epochs=None, n_iters=None, verbose=True):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        
        
        if n_iters is None and n_epochs is None:
            n_iters = 200   # default param for n_iters
        flag = np.isnan(train_data).all(axis=1)
        np.expand_dims(flag,1)
        train_data = train_data[~flag]
        
        train_loader = DataLoader(generate_data(train_data,train_labels,n_negatives=self.n_negatives,n_triplets=6400), batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for step,(data_a,data_p,data_negatives) in enumerate(train_loader,start=0):
                data_n = generate_negatives(data_negatives,self.n_negatives)# get the negatvie samples 
                if(self.aug_positives>0):
                    data_p = aug_data(data_a,data_p,self.aug_positives) # get all of the postivie samples: [positive,aug_data]
                    
                data_a,data_p,data_n = data_a.to(self.device),data_p.to(self.device),data_n.to(self.device)
                
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
               
              
            
               
                optimizer.zero_grad()
                train_1, train_2,train_3 = data_a.unsqueeze(2),data_p.unsqueeze(2),data_n.unsqueeze(2)
                out_a = self._net(train_1.to(self.device))
                out_p = self._net(train_2.to(self.device))
                out_n = self._net(train_3.to(self.device))
                
                
                
                loss = hierarchical_rank_loss(out_a, out_p,out_n,self.n_negatives,self.batch_size,self.number,self.distance,self.aug_positives,alpha=0.5, temporal_unit=0)
                
                  
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            logger.info(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            
            
        return loss_log
    
    def _eval_with_pooling(self, x):
        out = self.net(x.to(self.device, non_blocking=True))
        out = F.max_pool1d(out.transpose(1, 2),kernel_size = out.size(1),).transpose(1, 2)
        return out.cpu()
    
    def encode(self, data):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        data = data[..., np.newaxis]
        print("data_shape",data.shape)
       
        batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        #dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(data, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch
                out = self._eval_with_pooling(x)
                out = out.squeeze(1)
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
