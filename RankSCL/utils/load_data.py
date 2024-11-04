import os
import numpy as np 
import pandas 
# import seaborn as sns
import torch.nn.functional as F
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') 
def load_UCR_dataset(path, dataset):
    """
    Loads the UCR dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    train_file = os.path.join('/home/qir21001/AAAI2022/'+path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('/home/qir21001/AAAI2022/'+path, dataset, dataset + "_TEST.tsv")
    train_df = pandas.read_csv(train_file, sep='\t', header=None)
    test_df = pandas.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)
    
    '''
    train_labels, train = torch.Tensor(train_df[:,0]),torch.Tensor(train_df[:,1:])
    test_df = test_df.values
    test_labels, test = torch.Tensor(test_df[:,0]),torch.Tensor(test_df[:,1:])
    '''
    # Move the labels to {0,...,L-1}
    labels = np.unique(train_array[:, 0])
    print("Number of Classes:",labels.shape)
    # check if the data is 0-based
    transform = {}
    for i,l in enumerate(labels):
        transform[l]=i
    train = train_array[:,1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:,0])
    test = test_array[:,1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:,0])
    print("Train shape:",train.shape)
    print("Test shape:",test.shape)
    # Move the labels to {0, ..., L-1}
    

    
    

    # print(train_df.head())
    # train_labels = train_df[:][0].value_counts()
    # test_labels = test_df[:][0].value_counts()

    # print("Train value:",train_labels)
    # print("Test_value",test_labels)

    # datainfo = pandas.DataFrame({'Dataset':dataset,'Train_Shape':train_array.shape,'Test_shape':test_array.shape,'Class':labels.shape,'Train_labels':train_labels,'Test_labels':test_labels})
    # datainfo.to_csv('data_info.csv',index=False,sep=',')

    # pic = train_df.plot()
    # plt.show()
    # plt.savefig(dataset+'.jpg')


    # transform = {}
    # for i, l in enumerate(labels):
    #     transform[l] = i

    # train = numpy.expand_dims(train_array[:, 1:], 1).astype(numpy.float64)
    # train_labels = numpy.vectorize(transform.get)(train_array[:, 0])
    # test = numpy.expand_dims(test_array[:, 1:], 1).astype(numpy.float64)
    # test_labels = numpy.vectorize(transform.get)(test_array[:, 0])

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
         return train[...,np.newaxis], train_labels, test[...,np.newaxis], test_labels, labels.shape[0]
     
    # Post-publication note:
    # Using the testing set to normalize might bias the learned network,
    # but with a limited impact on the reported results on few datasets.
    # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
 

    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels, labels.shape[0]

def load_UEA(dataset):
    train_data = loadarff(f'/home/qir21001/AAAI2022/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'/home/qir21001/AAAI2022/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
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
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    if(np.isnan(train_X).any()):
        train_X = np.nan_to_num(train_X)
    if(np.isnan(test_X).any()):
        test_X = np.nan_to_num(test_X)
        
    return train_X, train_y, test_X, test_y, labels.shape[0]
# file_path = '../UCR'
# for i,j,k in os.walk(file_path):
#     dataset = i.split('/')[-1]
#     if(dataset !='UCR'):
#         print("data",dataset)
#         load_UCR_dataset('../UCR',dataset)