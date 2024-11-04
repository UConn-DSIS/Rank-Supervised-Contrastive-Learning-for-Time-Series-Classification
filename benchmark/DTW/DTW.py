import os
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.metrics import dtw
from Data.load_data import load_UCR_dataset,load_UEA
import argparse
import logging
import sys
from datetime import datetime
import torch
import numpy as np

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
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str,  required=True,
                        help='path where the dataset is located')
    
    parser.add_argument('--seed', type=int, required=True,
                        help='Set the seed for the experiment')


    return parser.parse_args()
if __name__ == '__main__':
    args = parse_arguments()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    
    logs_save_dir = 'DTW_results'
    experiment_log_dir = os.path.join('/home/qir21001/AAAI2022/',logs_save_dir, args.dataset)
    os.makedirs(experiment_log_dir, exist_ok=True)
    if(args.path == 'UCR'):
        X_train,y_train,X_test, y_test,nclasses = load_UCR_dataset(args.path, args.dataset)
      
    elif(args.path == 'UEA'):
        X_train,y_train, X_test, y_test,nclasses =load_UEA(args.dataset)
  
# logging 
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.dataset}')
    logger.debug("=" * 45)
    logger.debug("Data loaded----------")
    logger.debug(f"Seed:{args.seed}")
    knn = KNeighborsTimeSeriesClassifier(metric='dtw')
    param_grid = {'n_neighbors': [3, 5, 7]}
    train_size = X_train.shape[0]
    if  train_size //nclasses < 5 or train_size < 50:
        knn.fit(X_train, y_train)
    else:
        grid_search = GridSearchCV(knn, param_grid, scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)


        logger.debug("Best_Params:", grid_search.best_params_)
        logger.debug("Best_acc:%.3f", grid_search.best_score_)


        best_knn = grid_search.best_estimator_
    if  train_size //nclasses < 5 or train_size < 50:
        best_knn = knn
    test_predictions = best_knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    test_precision = precision_score(y_test, test_predictions, average='macro')
    test_f1 = f1_score(y_test, test_predictions, average='macro')

    logger.debug("Test Acc:%.3f", test_accuracy)
    logger.debug("Test Precision:%.3f", test_precision)
    logger.debug("Test F1:%.3f", test_f1)

   
