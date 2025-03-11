import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd

def set_path(data_config):
    base_path = data_config['base_path']
    aggregation_type = "subject-aggregated" if data_config['subject_agg'] == "sub_agg" else "subject-wise" if data_config['subject_agg'] == "sub_wise" else None
    mask_path = f"{base_path}/{aggregation_type}/masks/mask_{data_config['mask_num']}.npy"
    label_path = f"{base_path}/{aggregation_type}/raw_timeseries_label.npy"

    #! TO Ahhyun : added
    assert not (data_config['stft_type'] == "None" and data_config['norm_type'] == "None"), "both cannot be None"
    assert data_config['stft_type'] == "None" or data_config['norm_type'] == "None", "at least one should be None"
    
    if data_config['stft_type'] == "None":
        timeseries_type = "normalized_timeseries" if data_config['norm'] == "norm" else "raw_timeseries_data"
        if data_config['norm'] == "norm":
            data_path = f"{base_path}/{aggregation_type}/{timeseries_type}/{data_config['norm_type']}_timeseries_data.npy"
        else:
            data_path = f"{base_path}/{aggregation_type}/{timeseries_type}.npy"
    else:
        data_path = f"{base_path}/{aggregation_type}/stft/{data_config['stft_type']}_stft_timeseries_data-Zxx.npy"
        # TODO: Add logic to return additional STFT files if needed
    return data_path, label_path, mask_path

def collate_fn(batch, model_config):
    data, label = zip(*batch)
    data = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data]).unsqueeze(1)
    #TODO : EEGNet gets raw_timeseries, while others get STFT
    if model_config['model_name'] == 'EEGNet':
        data = data.unsqueeze(1)
        if model_config.get('downsampling_rate'):
            data = data[:, :, :, ::model_config['downsampling_rate']]
    label = torch.stack([torch.tensor(l, dtype=torch.long) for l in label])
    return data, label


class MyDataset(Dataset):
    def __init__(self, data_config, transform=None):
        self.transform = transform
        data_path, label_path, mask_path = set_path(data_config)
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.mask = np.load(mask_path)

    def init_io(self):
        self.info = pd.DataFrame(
            {
                "subject_id": [1],
                "trial_id": [1],
                "duration": [10000],
                "_record_id": ["_record_0"],
            }
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            sample_data = self.transform(sample_data)
        return sample_data, sample_label


def load_data(data_config, model_config):
    dataset = MyDataset(data_config)
    train_indices = np.where(dataset.mask == 0)[0]
    test_indices = np.where(dataset.mask == 1)[0]
    
    #further divide train_indices into final_train_indices and final_val_indices (take 30% as validation set)
    np.random.seed(42)
    np.random.shuffle(train_indices)
    print(train_indices[:10])
    final_train_indices = train_indices[int(len(train_indices)*0.3):] 
    final_val_indices = train_indices[:int(len(train_indices)*0.3)]
    

    # train_dataset = Subset(dataset, train_indices)
    train_dataset = Subset(dataset, final_train_indices)
    valid_dataset = Subset(dataset, final_val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=data_config.get('batch_size', 32), shuffle=True, collate_fn=lambda x: collate_fn(x, model_config)
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=data_config.get('batch_size', 32), shuffle=False, collate_fn=lambda x: collate_fn(x, model_config)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=data_config.get('batch_size', 32), shuffle=False, collate_fn=lambda x: collate_fn(x, model_config)
    )

    return train_loader, valid_loader, test_loader
