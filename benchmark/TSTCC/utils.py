import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


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


def plot_TSNE(data,data_label,file_path,nclasses):
    if(len(data.shape)>2):
        data = data.reshape(data.shape[0],-1)
    if not isinstance(data_label, np.ndarray):
        data_label = data_label.detach().cpu().numpy()
    if not isinstance(data,np.ndarray):
        data = data.detach().cpu().numpy()
    print(np.isnan(data))
    print(np.isfinite(data))
    tsne = TSNE(n_components=2,init='pca',random_state=42).fit_transform(data)
    x_min,x_max = tsne.min(0),tsne.max(0)
    x_norm = (tsne-x_min)/(x_max - x_min)
    df = pd.DataFrame()
    df['y'] = data_label
   

    print(data_label.shape)
    plt.figure(figsize=(8,8))
   
        #marker = class_marker_mapping[data_label[i]]
        
    plt.scatter(x_norm[:,0],x_norm[:,1],marker='o',c=data_label,cmap='rainbow',s=50)
   #     sc.set_facecolor("none")
  
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.show()
    plt.savefig(file_path+'.png',dpi=400)


def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("/home/qir21001/TS-TCC-main/main.py", os.path.join(destination_dir, "main.py"))
    copy("/home/qir21001/TS-TCC-main/trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"/home/qir21001/TS-TCC-main/config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("/home/qir21001/TS-TCC-main/dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("/home/qir21001/TS-TCC-main/dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"/home/qir21001/TS-TCC-main/models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("/home/qir21001/TS-TCC-main/models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("/home/qir21001/TS-TCC-main/models/TC.py", os.path.join(destination_dir, "TC.py"))
