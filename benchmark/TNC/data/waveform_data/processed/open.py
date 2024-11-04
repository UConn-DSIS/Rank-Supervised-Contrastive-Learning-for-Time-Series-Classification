import pickle
 
with open('/home/qir21001/TNC/data/waveform_data/processed/x_train.pkl','rb') as file:
     loaded_object = pickle.load(file)
print(loaded_object.shape)

with open("/home/qir21001/TNC/data/waveform_data/processed/state_train.pkl",'rb') as file:
    train_label = pickle.load(file)
print(train_label)
