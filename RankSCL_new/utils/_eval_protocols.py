import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from utils.utils import plot_TSNE
import torch
import torch.nn.functional as F

def fit_svm(features, y, logger,MAX_SAMPLES=10000):
    
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                    np.inf
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]
            
        grid_search.fit(features, y)
        #results = grid_search.cv_results_
    
        #for mean_score, params in zip(results["mean_test_score"], results["params"]):
        #    logger.info(f"Validation loss: {-mean_score:.3f} for {params}")
        
        return grid_search.best_estimator_

def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
        
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe

def fit_knn(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=1)
    )
    pipe.fit(features, y)
    return pipe

def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr

def eval_classification(model, model_name,train_data, train_labels, test_data, test_labels, dataset,logger,device,eval_protocol='linear'):
    device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    if(model_name == 'CNN'):
        if(train_data.ndim==2):
            train_repr = model(train_data.unsqueeze(1).float())
        else:
            train_repr = model(train_data.float())
    elif(model_name == 'LSTM'):
        train_data = train_data.permute(0,2,1)
        train_repr = model(train_data.float())
        #h = F.normzalize(h,dim=1)
        
    elif(model_name =='FCN'):
        if(train_data.ndim==2):
            _,train_repr = model(train_data.unsqueeze(1).float())
            print(train_repr.ndim)
            train_repr = F.normalize(train_repr,dim=1)
        else:
            _,train_repr = model(train_data.float())
            print(train_repr.ndim)
            train_repr = F.normalize(train_repr,dim=1)
            
    elif(model_name =='resnet'):
        data_1 = train_data.unsqueeze(1)
        train_repr =model(data_1.float())
        
    elif(model_name == 'dilation_CNN'):
        if(train_data.ndim==2):
            train_repr = model(train_data.unsqueeze(1).float())
        else:
            train_repr = model(train_data.float())
    elif(model_name == 'ts2vec'):
        train_repr = model.encode(train_data)
    
  
    print("train_labels_type",type(train_labels))
    print("train_repr_type",type(train_repr))
    #plot_TSNE(train_repr,train_labels,file_path='/home/qir21001/AAAI2022/TSNE/'+dataset+'_Encoder_repre',nclasses=nclasses)
    if(model_name == 'CNN'):
        test_repr = model(test_data.float())
    elif(model_name == 'LSTM'):
        test_data = test_data.permute(0,2,1)
        test_repr = model(test_data.float())
        #h = F.normzalize(h,dim=1)
        
    elif(model_name =='FCN'):
        if(test_data.ndim==2):
            _,test_repr = model(test_data.unsqueeze(1).float())
            test_repr = F.normalize(test_repr,dim=1)
        else: 
            _,test_repr = model(test_data.float())
            test_repr = F.normalize(test_repr,dim=1)
            
    elif(model_name =='resnet'):
        data_1 = test_data.unsqueeze(1)
        test_repr =model(data_1.float())
        
    elif(model_name == 'dilation_CNN'):
        if(test_data.ndim==2):
            test_repr = model(test_data.unsqueeze(1).float())
        else:
            test_repr = model(test_data.float())
    elif(model_name == 'ts2vec'):
        test_repr = model.encode(test_data)
   

    if eval_protocol == 'linear':
        fit_clf = fit_lr
    elif eval_protocol == 'svm':
        fit_clf = fit_svm
    elif eval_protocol == 'knn':
        fit_clf = fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)
    train_repr = train_repr.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()
    test_repr = test_repr.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()
    print(train_repr.shape)
    print(train_labels.shape)
    clf = fit_clf(train_repr, train_labels,logger)
   
    test_pred = clf.predict(test_repr)
    acc = clf.score(test_repr, test_labels)
    f1 =  f1_score(test_labels,test_pred,average="macro")
    precision = precision_score(test_labels,test_pred,average="macro")
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    
   # auprc = average_precision_score(test_labels_onehot, y_score)
   
    logger.debug('Test ACC %.3f,precision_score %.3f, f1_score%.3f',acc,precision,f1)
    return y_score, { 'acc': round(acc,3),'precision_score':round(precision,3),'f1_score':round(f1,3) }