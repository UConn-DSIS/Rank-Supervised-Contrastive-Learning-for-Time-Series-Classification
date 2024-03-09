import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from dataloader.sleep_study import change_ECG_value, normalize_data,generate_data
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
def load_UCR(dataset):
    train_file = os.path.join('/home/qir21001/TS2VEC/datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('/home/qir21001/TS2VEC/datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels,labels.shape[0]
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels,labels.shape[0]


def load_UEA(dataset):
    train_data = loadarff(f'/home/qir21001/TS2VEC/datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'/home/qir21001/TS2VEC/datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    '''
    创建一个StandardScaler对象，并调用fit方法对训练数据进行拟合。拟合过程计算并存储训练数据每个特征的均值和标准差。
    使用reshape方法将训练数据变形为二维数组，其中每行表示一个样本，每列表示一个特征。
    调用transform方法将变形后的训练数据进行标准化处理，通过减去均值并除以标准差来实现。标准化后的数据仍然是二维数组形式。
    使用reshape方法将标准化后的训练数据恢复为原始形状，与原始数据具有相同的维度和结构。
    对测试数据进行与训练数据相同的标准化处理，包括使用transform方法和reshape方法。
    通过这些步骤，训练数据和测试数据都被标准化为具有相同的尺度，使得它们在进行机器学习模型训练或预测时具有可比性和可解释性。标准化可以提高模型的收敛速度、减少特征之间的偏差，以及提高模型的鲁棒性。
    '''
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    '''
    这段代码的作用是将原始类别标签转换为它们在labels数组中的索引值，以便在后续的处理中更方便地使用。
    
    向量化操作是指将针对单个元素的操作扩展到整个数组或向量的操作。通常情况下，NumPy中的函数和操作符都支持对整个数组进行操作，这样可以通过并行计算和优化算法来获得更高效的计算。

例如，假设我们有两个相同大小的数组 a 和 b，我们想要计算它们对应位置的元素之和。使用向量化操作，我们可以直接执行 c = a + b，而不需要使用循环来逐个处理数组的每个元素。这样可以提高计算速度，并且代码更加简洁。

在NumPy中，很多函数和操作符都支持向量化操作，例如加法、减法、乘法、除法、求平方根等。这些操作会自动应用于整个数组，而不需要显式编写循环。

向量化操作的优势在于它利用了底层的优化机制，可以更高效地执行计算。同时，它也提供了更简洁和易读的代码，减少了编写和维护循环的复杂性。

总而言之，向量化操作是一种有效利用NumPy提供的功能和性能的方法，可以加速数组操作并简化代码。
    '''
    return train_X, train_y, test_X, test_y,labels.shape[0]
    
def load_sleep_study(dataset):
    ECG_1 = pd.read_csv(r'/home/qir21001/Sleep_Study/Subject '+str(dataset)+'/Sub_'+str(dataset)+'_PSGRawData_6_14_Part1.txt', engine='python',usecols=['Time','ECG_II'], header=0,sep=',')
    Pleth_1 = pd.read_csv(r'/home/qir21001/Sleep_Study/Subject '+str(dataset)+'/Sub_'+str(dataset)+'_PSGRawData_6_14_Part2.txt', engine='python',usecols=['Time','Pleth'], header=0,sep=',')
    label_1 = pd.read_csv(r'/home/qir21001/Sleep_Study/Subject '+str(dataset)+'/Subject '+str(dataset)+' PSG/SleepStaging'+str(dataset)+'.csv',engine='python',usecols=['Start Time ','Sleep Stage'],header=0,sep=',')
    print("loading finished----------------")
    ECG_new_1,time_1 = change_ECG_value(ECG_1,Pleth_1)
    ECG_new_1 = normalize_data(ECG_new_1)
    Pleth_new_1 = Pleth_1['Pleth'].values.reshape(-1,1)
    Pleth_new_1 = normalize_data(Pleth_new_1)
    data_sub1 = np.concatenate((time_1,ECG_new_1,Pleth_new_1),axis=1)
    X_1= generate_data(data_sub1[:,1:])
    X_1 = X_1.astype(float)
    X_filt_1 = X_1[~label_1['Sleep Stage'][:-1].eq('NS')]
    label_new_1 = label_1['Sleep Stage'].values.reshape(-1,1)
    label_new_1 = label_new_1[~label_1['Sleep Stage'].eq('NS')]
    label_bi_1 = pd.Series(label_new_1.flatten()).apply(lambda x: 0 if x == 'WK' else 1 if x == 'REM' else 2)
    label_bi_1 = np.array(label_bi_1).reshape(-1,1) 
    label_encoder = LabelEncoder()
    integer_label_1 = label_encoder.fit_transform(label_bi_1)
    training_size = int(len(X_filt_1) * 0.7)
    X_train, y_train = X_filt_1[:training_size], integer_label_1[:training_size]
    X_test, y_test = X_filt_1[training_size:], integer_label_1[training_size:]
    return X_train, y_train, X_test, y_test
    
class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, label, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset
        y_train = label
        if(np.isnan(X_train).any()):
            X_train = np.nan_to_num(X_train)
       
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        #if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.transpose(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(dataset, path,configs, training_mode):
    if(path == 'UCR'):
        train_dataset, train_labels, test_dataset, test_labels,nclasses= load_UCR(dataset)
    elif(path=='UEA'):
        train_dataset, train_labels, test_dataset, test_labels,nclasses= load_UEA(dataset)
 
     
    #train_set,val_set,train_label,val_label = train_test_split(train_dataset,train_labels,test_size=0.2,random_state=46,stratify=train_labels)
    train_set,val_set,train_label,val_label = train_test_split(train_dataset,train_labels,test_size=0.2,random_state=46)
    
    train_data = Load_Dataset(train_set,train_label,configs,training_mode)
    valid_data = Load_Dataset(val_set,val_label,configs,training_mode)
    test_data = Load_Dataset(test_dataset,test_labels,configs,training_mode)
    '''
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)
   '''
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader,nclasses,train_set.shape[1],train_set.shape[2]