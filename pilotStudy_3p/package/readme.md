# TorchEEG Model Runner

This package is designed to run various models based on TorchEEG. It supports running experiments using either a terminal command or a Python script and is tailored for EEG data in a "subject aggregated" or "single subject + 1 electrode" format.

## How to Run the Package

You can execute this package in two ways:

### 1. Using SLURM
Run the following command in the terminal:
```bash
sbatch sbatch_single.sh
```
This allows you to run single experiment with fixed configurations

### 2. Using Python
Make sure your current working directory is set to:
```
/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package
```
Activate conda environment:
```
conda activate /scratch/connectome/ahhyun724/.conda/envs/torcheeg
```
Run the following command:
```bash
python main.py --config
```

### 3. Using SLRUM, multi GPU
Run the following command in the terminal:
```bash
sbatch sbatch_multi.sh
```
This allows you to run multiple experiments looping over multiple configurations, distributing jobs in multiple gpus.
Make sure to adjust gpu_list, jobs_per_gpu when running this script

## Supported Features
- **Data Format**: The package only supports "subject aggregated" or "single subject + 1 electrode" EEG data.
- **Implemented Models**: The following models are implemented:
  - EEGNet (version 0; version 1 is included but untested)
  - Conformer
  - GRU
  - LSTM

## Model Configuration
- Variables for the models are defined in `model.py`.
- You can add new versions or configurations to experiment with different setups.

### EEGNet
- **Implemented Version**: Version 0
- **Version 1**: Included in `model.py` but has not been tested and may require debugging.

## Notes
1. Ensure that the current directory is correctly set before running the package.
2. Variables into models are defined in `model.py`, you can add new versions of model to try out different