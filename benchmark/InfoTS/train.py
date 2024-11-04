import numpy as np
import argparse
import time
import datetime
import datautils
from utils import init_dl_program
from infots import InfoTS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('dataset', help='The dataset name')
    # parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--archive', type=str, required=True, help='The archive name that the dataset belongs to. This can be set to UCR, UEA, forecast_csv, or forecast_csv_univar')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--log_file', type=str, default=None, help='log file')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')


    # hyper-parameters in backbone
    parser.add_argument('--batch-size', type=int, default=32, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=int, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--beta', type=float, default=1.0, help='trade off between local and global contrastive')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=400, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='The probility of dropout')
    parser.add_argument('--split_number', type=int, default=8, help='split_number in local contrastive')


    # hypaer-parameters in meta-learner and augmentations
    parser.add_argument('--label_ratio', type=float, default=1.0, help='Number of labels in training set to train meta-learner')
    parser.add_argument('--meta_beta', type=float, default=0.4, help='The probility of meta_lambda')
    parser.add_argument('--aug_p1', type=float, default=0.4, help='The probility of augmentation 1')
    parser.add_argument('--aug_p2', type=float, default=0., help='The probility of augmentation 2') # zero means use the oringinal input.
    parser.add_argument('--meta_lr', type=int, default=0.001, help='The learning rate (defaults to 0.01)')
    parser.add_argument('--supervised_meta', action="store_true", help='meta in supervised setting and unsupervised setting')

    args = parser.parse_args()

    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    if args.archive == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
       
    elif args.archive == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        print("train_data_shape",train_data.shape)
        print("train_label_shape",train_labels.shape)
    elif args.archive == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
    else:
        raise ValueError(f"Archive type {args.archive} is not supported.")
    if task_type == 'classification':
        B, T, D = train_data.shape
        print("Dataset: %s  #instance %d  #dim  %d  #leng %d  #label %d" % (args.dataset,B,T,D,len(np.unique(train_labels))))


        if args.label_ratio < 1.0:
            train_labels_e = np.expand_dims(np.expand_dims(train_labels, 1), 1)
            train_data_labels = np.concatenate([train_data, train_labels_e], 1)
            np.random.shuffle(train_data_labels)
            nmb_train = int(args.label_ratio * train_data.shape[0])
            train_data_label_for_validation = train_data_labels[:nmb_train]
            train_data_for_validation = train_data_label_for_validation[:, :-1, :]
            train_label_for_validation = train_data_label_for_validation[:, -1, 0]
        else:
            train_data_for_validation = train_data
            train_label_for_validation = train_labels
        valid_dataset = (train_data_for_validation, train_label_for_validation, test_data, test_labels)
    elif task_type == 'forecasting':
        valid_dataset = (data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)

    if train_data.shape[0]==1:
        train_slice_number = int(train_data.shape[1]/args.max_train_length)
        if train_slice_number<args.batch_size:
            args.batch_size = train_slice_number
    else:
        if train_data.shape[0]<args.batch_size:
            args.batch_size = train_data.shape[0]
    print("Arguments:", str(args))



    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        meta_lr = args.meta_lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )

    t = time.time()
    
    model = InfoTS(
        input_dims=train_data.shape[-1],
        device=device,
        num_cls = len(np.unique(train_labels)) if args.supervised_meta and task_type == 'classification' else args.batch_size,
        dropout = args.dropout,
        aug_p1= args.aug_p1,
        aug_p2 = args.aug_p2,
        **config
    )

    res = model.fit(train_data,
         task_type = task_type,
         meta_beta=args.meta_beta,
         n_epochs=args.epochs,
         n_iters=args.iters,
         beta = args.beta,
         verbose=False,
         miverbose=True,
         split_number=args.split_number,
         supervised_meta = args.supervised_meta if task_type == 'classification' else False, # for forecasting, use unsupervised setting.
         valid_dataset = valid_dataset,
         train_labels=train_labels if args.supervised_meta and task_type == 'classification' else None
        )
    if task_type == 'classification':
        loss_log, acc_log, precision_log, f1_log = res
        acc = np.mean(acc_log[-5:])
        precision = np.mean(precision_log[-5:])
        f1 =  np.mean(f1_log[-5:])
        mi_info = 'acc %.3f preicison %.3f f1  %.3f' % (acc,precision,f1)
    else:
        mse,mae = res
        mi_info = 'mse %.5f  mae%.5f' % (mse[-1],mae[-1])

    print(mi_info)

    if args.log_file is not None:
        with open(args.log_file,'a') as fout:
            fout.write(args.dataset+'\t super='+str(args.supervised_meta)+mi_info+'\n')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print("Finished.")

