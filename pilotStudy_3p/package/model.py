# model.py
import torch
import torcheeg
from torcheeg.models import EEGNet, Conformer, GRU, LSTM

def get_model(model_config):
    model_name = model_config['model_name']
    if model_name == "EEGNet":
        model = EEGNet_ret(model_config)
    elif model_name == "Conformer":
        model = Conformer(num_electrodes=150,
                  sampling_rate=500,
                  hid_channels=40,
                  depth=6,
                  heads=10,
                  dropout=0.5,
                  forward_expansion=4,
                  forward_dropout=0.5,
                  num_classes=2)
    elif model_name == "GRU":
        model = GRU(num_electrodes=1, hid_channels=64, num_classes=2)
    elif model_name == "LSTM":
        model = LSTM(num_electrodes=1, hid_channels=64, num_classes=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def EEGNet_ret(model_config):
    EEGNet_vars = [{
        'kernel_1': 64,
        'kernel_2': 16,
        'F1': 8,
        'F2': 16,
        'D': 2,
        'num_classes': 2
    }, {
        'kernel_1': 128,
        'kernel_2': 8,
        'F1': 8,
        'F2': 16,
        'D': 2,
        'num_classes': 2
    }]
    model_ver = model_config.get('model_ver', 0)
    if not (0 <= model_ver < len(EEGNet_vars)):
        raise ValueError(f"Invalid model_ver: {model_ver}. Must be 0 or 1.")
    
    EEGNet_var=EEGNet_vars[model_ver]
    
    EEGNet_model = EEGNet(chunk_size=int(10000 / model_config.get('downsampling_rate', 1)) , #default downsampling_rate=1
                    num_electrodes=1,
                    dropout=model_config['dropout'],
                    kernel_1=EEGNet_var['kernel_1'],
                    kernel_2=EEGNet_var['kernel_2'],
                    F1=EEGNet_var['F1'],
                    F2=EEGNet_var['F2'],
                    D=EEGNet_var['D'],
                    num_classes=2)
    return EEGNet_model


## TODO : Add Conformer_ret, GRU_ret, LSTM_ret functions + update get_model function