import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import TensorDataset, DataLoader
from models import TSEncoder
from scipy.special import softmax
from models.losses import *
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.metrics import log_loss
import tasks
from models.basicaug import *
from models.augmentations import AutoAUG
LAEGE_NUM = 1e7
import nni

class InfoTS:
    '''The InfoTS model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=60,
        num_cls = 2,
        depth=10,
        device='cuda',
        lr=0.001,
        meta_lr = 0.01,
        batch_size=16,
        max_train_length=None,
        mask_mode = 'binomial',
        dropout = 0.1,
        aug_p1=0.2,
        aug_p2=0.0,
        t0=2.0,
        t1 = 0.1,
        eval_every_epoch = 20,
        used_augs = None,
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            meta_lr (int): The learning rate for meta learner.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims,
                                  hidden_dims=hidden_dims, depth=depth,
                                  dropout=dropout,mask_mode=mask_mode).to(self.device)

        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.n_epochs = 0
        self.n_iters = 0

        self.pred = torch.nn.Linear(output_dims,num_cls).cuda()
        self.unsup_pred = torch.nn.Linear(output_dims,batch_size).cuda()
        self.t0 = 2.0
        self.t1 = 0.1

        self.aug = AutoAUG(aug_p1=aug_p1,aug_p2=aug_p2,used_augs=used_augs).cuda()
        self.meta_lr = meta_lr

        # contrarive between aug and original
        self.single = (aug_p2==0.0)
        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.cls_lr = meta_lr
        self.eval_every_epoch = eval_every_epoch

    def get_dataloader(self,data,shuffle=False, drop_last=False):

        # pre_process to return data loader
        '''
        if self.max_train_length is not None:
            sections = data.shape[1] // self.max_train_length
            if sections >= 2:
                data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)
    '''
        temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            data = centerize_vary_length_series(data)

        data = data[~np.isnan(data).all(axis=2).all(axis=1)]
        data = np.nan_to_num(data) # change nan to 0
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)),shuffle=shuffle, drop_last=drop_last)
        return data, dataset, loader


    def get_features(self,x,n_epochs=None):
        if n_epochs==None:
            t =1.0
        else:
            t = float(self.t0 * np.power(self.t1 / self.t0, (self.n_epochs+1) / n_epochs))
        # t = 1.0
        a1,a2 = self.aug((x,t))

        if torch.isnan(a1).any() or torch.isnan(x).any():
            exit(1)



        out1 = self._net(a1)
        out2 = self._net(a2)
        return out1,out2


    # calculate mutual information MI(v,x)
    def MI(self, data_loader):
        ori_training = self._net.training
        self._net.eval()
        cum_vx = 0
        zvs = []
        zxs = []
        size = 0
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)
                outv, outx = self.get_features(x)
                vx_infonce_loss = L1out(outv, outx) * x.size(0)
                size +=x.size(0)

                zv = F.max_pool1d(outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)).transpose(1,2).squeeze(1)
                zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1,2).squeeze(1)

                cum_vx += vx_infonce_loss.item()
                zvs.append(zv.cpu().numpy())
                zxs.append(zx.cpu().numpy())

        MI_vx_loss = cum_vx / size
        zvs = np.concatenate(zvs,0)
        zxs = np.concatenate(zxs,0)

        if ori_training:
            self._net.train()
        return zvs,MI_vx_loss

    def fit(self, train_data, n_epochs=None, n_iters=None,task_type='classification' ,verbose=False,supervised_meta=True,beta=1.0,\
            valid_dataset=None, miverbose=None, split_number=8,meta_epoch=2,meta_beta=1.0,train_labels = None):
        ''' Training the InfoTS model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            supervised_meta (bool) Whether to use label for training meta-learner.
            beta (float): trade-off between global and local contrastive.
            valid_dataset:  (train_data, train_label,test_data,test_label) for Classifier.
            miverbose (bool): Whether to print the information of meta-learner
            meta_epoch (int): meta-parameters are updated every meta_epoch epochs
            meta_beta (float): trade-off between high variety and high fidelity.
            task_type (str): downstream task
            train_labels: The label of training data, used for the supervised setting.
        Returns:
            crietira.
        '''

        # check the input formation
        assert train_data.ndim == 3

        # default param for n_iters
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        train_data,train_dataset,train_loader =  self.get_dataloader(train_data,shuffle=True,drop_last=True)
        print("train_dataset:",len(train_dataset))
        print("train labels:",len(train_labels))
        cls_optimizer  = None

        if not supervised_meta:
            train_labels = TensorDataset(torch.arange(train_data.shape[0]).to(torch.long).cuda())
            cls_optimizer = torch.optim.AdamW(self.unsup_pred.parameters(), lr=self.cls_lr)

        else:
            train_labels = TensorDataset(torch.from_numpy(train_labels).to(torch.long).cuda())
            cls_optimizer = torch.optim.AdamW(self.pred.parameters(), lr=self.cls_lr)

        train_data_label = []
       
        for i in range(len(train_dataset)):
            train_data_label.append([train_dataset[i], train_labels[i]])
        train_data_label_loader = DataLoader(train_data_label, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        if task_type=='classification' and valid_dataset is not None:
            cls_train_data, cls_train_labels, cls_test_data, cls_test_labels = valid_dataset
            cls_train_data,cls_train_dataset,cls_train_loader = self.get_dataloader(cls_train_data,shuffle=False,drop_last=False)


        meta_p = self.aug.parameters()
        meta_optimizer = torch.optim.AdamW(meta_p, lr=self.meta_lr)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)


        acc_log = []
        vy_log = []
        vx_log = []
        loss_log = []
        f1_log = []
        precision_log = []

        mses = []
        maes = []

        def eval(final=False):
            self._net.eval()
            if task_type == 'classification':
                out, eval_res = tasks.eval_classification(self, cls_train_data, cls_train_labels, cls_test_data,
                                                          cls_test_labels, eval_protocol='svm')
                clf = eval_res['clf']
                zvs, MI_vx_loss = self.MI(cls_train_loader)
                
                v_pred = softmax(clf.decision_function(zvs), -1)
                MI_vy_loss = log_loss(cls_train_labels, v_pred)
                v_acc = clf.score(zvs, cls_train_labels)

                vx_log.append(MI_vx_loss)
                vy_log.append(MI_vy_loss)

                acc_log.append(eval_res['acc'])
                
                f1_log.append(eval_res['f1'])
                precision_log.append(eval_res['precision'])
                # try:
                #nni.report_intermediate_result(eval_res['acc'])
                # except:
                #     pass
                if miverbose:
                    print('acc %.3f precision %.3f F1 %.3f' % (
                    eval_res['acc'], eval_res['precision'],eval_res['f1']))
            elif task_type == 'forecasting':
                if not final:
                    valid_dataset_during_train = valid_dataset[0],valid_dataset[1],valid_dataset[2],valid_dataset[3],valid_dataset[4],[valid_dataset[5][0]],valid_dataset[6]
                    out, eval_res = tasks.eval_forecasting(self, *valid_dataset_during_train)
                else:
                    out, eval_res = tasks.eval_forecasting(self, *valid_dataset)

                res = eval_res['ours']
                mse = sum([res[t]['norm']['MSE'] for t in res]) / len(res)
                mae = sum([res[t]['norm']['MAE'] for t in res]) / len(res)
                mses.append(mse)
                maes.append(mae)
                nni.report_intermediate_result(mse + mae)
                print(eval_res['ours'])
                return eval_res['ours']

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            if (self.n_epochs + 1) % meta_epoch == 0:
                self.meta_fit(train_data_label_loader, meta_optimizer,meta_beta,supervised_meta,cls_optimizer)

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            self._net.train()

            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                optimizer.zero_grad()
                meta_optimizer.zero_grad()

                out1,out2 = self.get_features(x,n_epochs=n_epochs)

                loss = global_infoNCE(out1, out2) + local_infoNCE(out1, out2, k=split_number)*beta

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1


            self.n_epochs += 1
            if self.n_epochs%self.eval_every_epoch==0:
                eval()


            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
                print(self.aug._parameters)
        res = eval(final=True)
        if task_type == 'classification':
            return loss_log,acc_log,precision_log,f1_log
        else:
            return [res[t]['norm']['MSE'] for t in res] ,[res[t]['norm']['MAE'] for t in res],mses[-1],maes[-1]

    def meta_fit(self, train_loader,meta_optimizer,meta_beta,supervised_meta,cls_optimizer):
        pre_flag = self._net.training
        self._net.eval()
        for batch in train_loader:
            x = batch[0][0]
            y = batch[1][0]

            if self.max_train_length is not None and x.size(1) > self.max_train_length:
                window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                x = x[:, window_offset: window_offset + self.max_train_length]
            x = x.to(self.device)
            if supervised_meta:
                y = y.to(self.device)
            else:
                y = torch.arange(self.batch_size,dtype=torch.int64).to(self.device)

            meta_optimizer.zero_grad()
            outv, outx = self.get_features(x)
            MI_vx_loss = L1out(outv, outx) #minimize v1v2_MI maximize v1v2_loss


            zv = F.max_pool1d(outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)).transpose(1, 2)
            zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1, 2)

            if supervised_meta:
                pred_yv = torch.squeeze(self.pred(zv),1)
                pred_yx = torch.squeeze(self.pred(zx),1)

            else:
                pred_yv = torch.squeeze(self.unsup_pred(zv),1)
                pred_yx = torch.squeeze(self.unsup_pred(zx),1)


            MI_vy_loss = self.CE(pred_yv, y)
            MI_xy_loss = self.CE(pred_yx, y)
            # MI_vx_ce = self.BCE(pred_yx,pred_yx)

            # minimize MI_vx Maximize MI_vy -> minimize  MI_vx-beta*MI_vy -> minimize -MI_vx_loss+beta*MI_vy_loss
            vx_vy_loss = -MI_vx_loss+meta_beta*(MI_vy_loss+MI_xy_loss)
            # print(self.n_epochs,MI_vx_loss,MI_vy_loss,MI_xy_loss,self.aug.weight)

            vx_vy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.aug.parameters(), 2.0)
            meta_optimizer.step()
            # MI_xy_loss.backward()
            cls_optimizer.step()


        # self.super_fit(train_loader, supervised_meta, cls_optimizer)
        if pre_flag:
            self._net.train()


    def encode(self, data, mask=None, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        Returns:
            repr: The representations for data.
        '''


        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self.net(x.to(self.device, non_blocking=True), mask)
                out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).transpose(1, 2).cpu()
                out = out.squeeze(1)

                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()

    def casual_encode(self, data, encoding_window=None, mask=None, sliding_length=None, sliding_padding=0,  batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        Returns:
            repr: The representations for data.
        '''
        casual = True
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
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

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()


    def super_fit(self, train_loader,supervised_meta, cls_optimizer):

        for batch in train_loader:
            x = batch[0][0]
            y = batch[1][0]

            if self.max_train_length is not None and x.size(1) > self.max_train_length:
                window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                x = x[:, window_offset: window_offset + self.max_train_length]
            x = x.to(self.device)
            if supervised_meta:
                y = y.to(self.device)
            else:
                y = torch.arange(self.batch_size, dtype=torch.int64).to(self.device)

            cls_optimizer.zero_grad()
            outx = self._net(x)
            zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1, 2)

            if supervised_meta:
                pred_yx = torch.squeeze(self.pred(zx), 1)
            else:
                pred_yx = torch.squeeze(self.unsup_pred(zx), 1)

            celoss = self.CE(pred_yx, y)
            celoss.backward()
            cls_optimizer.step()
