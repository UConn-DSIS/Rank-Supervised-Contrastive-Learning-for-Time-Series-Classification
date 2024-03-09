import pandas as pd
import numpy as np 


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
def change_ECG_value(ECG_1,Pleth_1):
  ECG = ECG_1['ECG_II'].values.reshape(-1,1)
  print(ECG.shape)
  ECG_new_1 = np.mean(ECG.reshape(-1, 2, ECG.shape[1]), axis=1)
  print(ECG_new_1)
  print(ECG_new_1.shape)
  time = Pleth_1['Time']
  time = time.astype(str).values
  print(type(time))
  time_split = np.char.split(time.astype(str))  
  time_split = np.stack(time_split,axis=0)
  print(time_split)
  time_new = []
  time = time_split[:,1]
  for i in range(len(time)):
    if(int(time[i][:2]) > 12):
      hour = int(time[i][:2]) - 12
      time[i] = str(hour)+time[i][2:]+' PM'
    elif(time[i][0]=='0'):
      time[i] = time[i][1:]+' AM'
    else:  
      time[i] = time[i]+' AM'
  return ECG_new_1, time.reshape(-1,1)

def normalize_data(data):
     
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data
  
def generate_data(X, sequence_length = 3000, step = 3000):
    X_local = []
   
    for start in range(0, len(X) - sequence_length, step):
        end = start + sequence_length
        X_local.append(X[start:end])
   
    return np.array(X_local)

 # load data
