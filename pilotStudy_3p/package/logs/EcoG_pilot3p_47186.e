/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package/utils.py:28: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  data = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data]).unsqueeze(1)
/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package/utils.py:34: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  label = torch.stack([torch.tensor(l, dtype=torch.long) for l in label])
