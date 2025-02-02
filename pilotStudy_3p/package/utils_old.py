# utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
import pandas as pd


def set_path(data_config):
    base_path = data_config['base_path']
    if data_config['subject_agg'] == 'sub_agg':
        mask_path = f"{base_path}/subject-aggregated/masks/mask_{data_config['mask_num']}.npy"
        label_path = f"{base_path}/subject-aggregated/raw_timeseries_labels.npy"
        if data_config['stft_type'] == "None":
            if data_config['norm'] == 'norm':
                data_path = f"{base_path}/subject-aggregated/normalized_timeseries/{data_config['norm_method']}_timeseries_data.npy"
            else: #data_config['norm'] == 'raw':
                data_path = f"{base_path}/subject-aggregated/raw_timeseries_data.npy"
        else: #data_config['stft'] == some method
            data_path = f"{base_path}/subject-aggregated/stft/{data_config['stft_type']}_stft_timeseries_data-Zxx.npy"
            
    else: # sub_wise
        mask_path = f"{base_path}/subject-wise/masks/mask_{data_config['mask_num']}.npy"
        label_path = f"{base_path}/subject-wise/raw_timeseries_labels.npy"
        if data_config['stft_type'] == "None":
            if data_config['norm'] == 'norm':
                data_path = f"{base_path}/subject-wise/normalized_timeseries/{data_config['norm_method']}_timeseries_data.npy"
            else: #data_config['norm'] == 'raw':
                data_path = f"{base_path}/subject-wise/raw_timeseries_data.npy"
        else: #data_config['stft'] == some method
            data_path = f"{base_path}/subject-wise/stft/{data_config['stft_type']}_stft_timeseries_data-Zxx.npy"
            #TODO: implement function to return stft_timeseries_data-f.npy, stft_timeseries_data-t.npy
    return data_path, label_path, mask_path

def collate_fn(batch, model_config):
    data, label = zip(*batch)
    data = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data]).unsqueeze(1)  # Add electrode dimension
    if model_config['name'] == 'EEGNet':
        data = data.unsqueeze(1) #only for EEGNet #TODO : check if needed for EEGConformer
        data = data[:,:,:,::4] #TODO : implement downsampling option
    label = torch.stack([torch.tensor(l, dtype=torch.long) for l in label])
    return data, label


class MyDataset(Dataset, data_config):
    def __init__(self, data_config, transform=None):
        data_path, label_path, mask_path = set_path(data_config)
        data = np.load(data_path)
        labels = np.load(label_path)
        mask = np.load(mask_path)
        
    def init_io(self): #Dummy function to match torchEEG API
        self.info = pd.DataFrame({
                "subject_id": [1],
                "trial_id": [1],
                "duration": [10000],
                "_record_id": ["_record_0"]
            })
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)  # Ensure data is float32
        sample_label = torch.tensor(self.label[idx], dtype=torch.float32)  # Ensure label is float32
        
        if self.transform:
            sample_data = self.transform(sample_data)
        
        return sample_data, sample_label




def load_data(data_config):
    data_path, label_path, mask_path = set_path(data_config)
    data = np.load(data_path)
    labels = np.load(label_path)
    mask = np.load(mask_path)
    
    # Create train and test indices based on the mask
    train_indices = np.where(mask == 0)[0]
    test_indices = np.where(mask == 1)[0]
    
    # Create train and test subsets
    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)

    # Create DataLoader for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader
