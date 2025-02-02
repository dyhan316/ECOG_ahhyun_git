Traceback (most recent call last):
  File "/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package/main.py", line 87, in <module>
    model = train_model(config , train_loader, test_loader)
  File "/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package/train.py", line 12, in train_model
    model = get_model(config['model'])
  File "/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package/model.py", line 9, in get_model
    model = EEGNet_ret(model_config)
  File "/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package/model.py", line 50, in EEGNet_ret
    EEGNet_model = EEGNet(chunk_size=10000 / model_config.get('downsampling_rate', 1) , #default downsampling_rate=1
  File "/scratch/connectome/ahhyun724/DIVER/torcheeg/torcheeg/torcheeg/models/cnn/eegnet.py", line 110, in __init__
    self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)
  File "/scratch/connectome/ahhyun724/DIVER/torcheeg/torcheeg/torcheeg/models/cnn/eegnet.py", line 114, in feature_dim
    mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
TypeError: zeros(): argument 'size' failed to unpack the object at pos 4 with error "type must be tuple of ints,but got float"
