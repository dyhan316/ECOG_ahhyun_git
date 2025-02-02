import numpy as np 
from utils import load_data, MyDataset
import yaml
from torch.utils.data import Subset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# config = load_config('./config.yaml')
config = load_config('./test.yaml')

aa = MyDataset(config['data'])

def get_train_test_split(data, label, mask) : 
    train_indices = np.where(mask == 0)[0]
    test_indices = np.where(mask == 1)[0]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    train_label = label[train_indices]
    test_label = label[test_indices]
    
    return train_data, test_data, train_label, test_label


def get_train_test_data(data_config) : 
    dataset = MyDataset(data_config)
    train_data, test_data, train_label, test_label = get_train_test_split(dataset.data, dataset.labels, dataset.mask)
    
    #flatten the last two dims if STFT 
    #only for ML as we flatten stuff 
    if data_config['stft_type'] != "None":
        #may not work if the shape is not (n, m, l)
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)    
    
    return train_data, test_data, train_label, test_label
import pdb ; pdb.set_trace()
train_data, test_data, train_label, test_label = get_train_test_data(config['data'])
