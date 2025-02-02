#!/bin/bash
#SBATCH --job-name=EcoG_pilot3p 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=2
#SBATCH --nodelist=node3
#SBATCH --qos interactive
#SBATCH --time 1:00:00
#SBATCH -e ./logs/EcoG_pilot3p_%j.e
#SBATCH -o ./logs/EcoG_pilot3p_%j.o

# Load necessary modules
conda init bash
source activate /scratch/connectome/ahhyun724/.conda/envs/torcheeg

basedir="/scratch/connectome/ahhyun724/DIVER/torcheeg/pilotStudy_3p/package"
cd $basedir || { echo "Base directory $basedir does not exist! Exiting..."; exit 1; }


# Prepare for experiments
TZ='Asia/Seoul'; export TZ
start=$(date +%s)
pwd; hostname; date

# Set variables
## TRAIN
batch_size=32
epochs=20
learning_rate=0.001
weight_decay=0.005
criterion="CrossEntropyLoss"  # Options: "CrossEntropyLoss", "BCEWithLogitsLoss" "MSELoss"
optimizer="Adam"  # Options: "Adam", "SGD", "RMSprop", "Adagrad"
## MODEL
model_name="EEGNet"
model_ver=0
downsampling_rate=4
num_classes=2
## DATA
dataset_name="3ppl" #fixed
base_path="/scratch/connectome/dyhan316/ECOG_PILOT/data_rearranged"
mask_num=0 #select btw [0,49]
subject_agg="sub_agg" # "sub_agg", "sub_wise"
norm="norm" #"norm", "raw"
norm_type="fixation_denormed" #"fixation_denormed", "fixation_normalized", "whole_trial_denormed", "whole_trial_normalized"  "None"(should be None if norm is "raw")
stft_type="None" # "base" "halfnfft" "halfoverlap_halfnfft" "nooverlap_halfnfft" "None"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "main.py not found in $basedir. Exiting..."
    exit 1
fi



func() {
    echo "Running experiment..."
    python main.py --config config.yaml \
    --batch_size $batch_size --epochs $epochs --learning_rate $learning_rate --weight_decay $weight_decay\
    --criterion $criterion --optimizer $optimizer \
    --model_name $model_name --model_ver $model_ver --downsampling_rate $downsampling_rate --num_classes $num_classes \
    --dataset_name $dataset_name \
    --base_path $base_path --mask_num $mask_num \
    --subject_agg $subject_agg --norm $norm --norm_type $norm_type --stft_type $stft_type
}

#python main.py --config config.yaml     --batch_size 32 --epochs 20 --learning_rate 0.001 --weight_decay 0.005    --criterion CrossEntropyLoss --optimizer Adam     --model_name EEGNet --model_ver 0 --downsampling_rate 4 --num_classes 2     --dataset_name 3ppl     --base_path /scratch/connectome/dyhan316/ECOG_PILOT/data_rearranged --mask_num 0     --subject_agg sub_agg --norm norm --norm_type fixation_denormed --stft_type None


# Run your application
func


# Print time spent & end date
end=$(date +%s)
echo experiment ended
date
duration=$((end - start))
echo "Elapsed time: $(($duration / 3600)) hours, $(($duration / 60 % 60)) minutes, $(($duration % 60)) seconds"


exit 0