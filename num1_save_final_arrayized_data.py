import os 
import scipy.io as sio
import numpy as np
import pandas as pd

DATA_BASE_PATH = './Data_Preprocessed'

#dict of format "Subject" : ("Preprocessed Data", "Success Data (label)")
DATA_LABEL_TUPLE_DICT = {"JHJ": ("JHJ_Prepro.mat","JHJ_Success.mat") ,
                         "LDH" : ("LDH_Prepro.mat","LDH_Success.mat") ,
                         "LSJ" : ("LSJ_Prepro.mat","LSJ_Success.mat") }
DATA_SUBTPYE_NAME_LIST = ['E1_Target', 'E2_Target', 'E3_Target', 'E4_Target', 'E5_Target', 'E6_Target']

def load_data_and_labels(data_base_path, data_label_tuple):
    data_path = os.path.join(data_base_path, data_label_tuple[0])
    label_path = os.path.join(data_base_path, data_label_tuple[1])
    data = sio.loadmat(data_path)
    label = sio.loadmat(label_path)
    return data, label

####=====codes SPECFIIC TO MEMORY TASK's method of saving the data =====####
def arraize_data(data):
    data_array = []
    for subtype in DATA_SUBTPYE_NAME_LIST:
        data_array.append(split_into_5s_intervals(data[subtype]))
    return np.array(data_array)
    
def split_into_5s_intervals(data):
    #assuems 2kHz sampling rate
    size = 2000*5
    return np.array([data[:,i:i+size] for i in range(0, data.shape[1], size)])
    
def load_arrayized_data_and_labels(data_base_path, data_label_tuple_dict):
    total_data_array = []
    total_label_array = []
    for sub, data_label_tuple in data_label_tuple_dict.items():
        data, label = load_data_and_labels(data_base_path, data_label_tuple)
        data_array = arraize_data(data)
        label_array = label['Success'].T
        total_data_array.append(data_array)
        total_label_array.append(label_array)
    total_data_array = np.array(total_data_array)
    total_label_array = np.array(total_label_array)
    #shape (3, 6, 20, 1, 10000) (3, 6, 20) #(subject, subtype, trial, channel, time), (subject, subtype, label)
    return total_data_array, total_label_array

###!!TODO : must implement splitting into 5s intervals!


#try running
final_data_array, final_label_array = load_arrayized_data_and_labels(DATA_BASE_PATH, DATA_LABEL_TUPLE_DICT)
print(final_data_array.shape, final_label_array.shape)  
#save
np.save('./data_rearranged/final_data_array.npy', final_data_array)
np.save('./data_rearranged/final_label_array.npy', final_label_array)





#split by taking one from each or sth! 