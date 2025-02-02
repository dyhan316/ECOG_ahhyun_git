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

# Define parameter lists
learning_rate_list=(0.0001 0.001 0.01)
weight_decay_list=(0.001 0.01)
model_ver_list=(0) #(0 1)
downsampling_rate_list=(4 2)

mask_num_list=(0 10 20 30 40 49)
subject_agg_list=("sub_agg") #("sub_agg" "sub_wise"), only "sub_agg" is used in this study
norm_type_list=("fixation_denormed" "fixation_normalized" "whole_trial_denormed" "whole_trial_normalized") #"None"
stft_type_list=("base" "halfnfft" "halfoverlap_halfnfft" "nooverlap_halfnfft") #"None"

gpu_list=(0 1)  # GPUs available for jobs
jobs_per_gpu=2  # Number of jobs to run per GPU


## TRAIN
batch_size=32
epochs=20
# learning_rate=0.001
# weight_decay=0.005
criterion="CrossEntropyLoss"  # Options: "CrossEntropyLoss", "BCEWithLogitsLoss" "MSELoss"
optimizer="Adam"  # Options: "Adam", "SGD", "RMSprop", "Adagrad"
## MODEL
model_name="EEGNet"
# model_ver=0
# downsampling_rate=4
num_classes=2
## DATA
dataset_name="3ppl" #fixed
base_path="/scratch/connectome/dyhan316/ECOG_PILOT/data_rearranged"
# mask_num=0 #select btw [0,49]
# subject_agg="sub_agg" # "sub_agg", "sub_wise"
norm="norm" #"norm", "raw"
# norm_type="fixation_denormed" #"fixation_denormed", "fixation_normalized", "whole_trial_denormed", "whole_trial_normalized"  "None"(should be None if norm is "raw")
# stft_type="None" # "base" "halfnfft" "halfoverlap_halfnfft" "nooverlap_halfnfft" "None"



# Ensure main.py exists
if [ ! -f "main.py" ]; then
    echo "main.py not found in $basedir. Exiting..."
    exit 1
fi

# Function to run experiments
run_experiment() {
    local learning_rate=$1
    local weight_decay=$2
    local model_ver=$3
    local downsampling_rate=$4
    local mask_num=$5
    local subject_agg=$6
    local norm_type=$7
    local stft_type=$8
    local gpu_id=$9

    echo "Running experiment: lr=$learning_rate, wd=$weight_decay, ver=$model_ver, dsr=$down_sampling_rate, mask=$mask_num, agg=$subject_agg, norm=$norm_type, stft=$stft_type on GPU $gpu_id..."
   
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --config config.yaml \
    --batch_size $batch_size --epochs $epochs --learning_rate $learning_rate --weight_decay $weight_decay\
    --criterion $criterion --optimizer $optimizer \
    --model_name $model_name --model_ver $model_ver --downsampling_rate $downsampling_rate --num_classes $num_classes \
    --dataset_name $dataset_name \
    --base_path $base_path --mask_num $mask_num \
    --subject_agg $subject_agg --norm $norm --norm_type $norm_type --stft_type $stft_type
}

# Distribute jobs across GPUs with multiple jobs per GPU
gpu_idx=0
job_count=0

for learning_rate in "${learning_rate_list[@]}"; do
    for weight_decay in "${weight_decay_list[@]}"; do
        for model_ver in "${model_ver_list[@]}"; do
            for downsampling_rate in "${downsampling_rate_list[@]}"; do
                for mask_num in "${mask_num_list[@]}"; do
                    for subject_agg in "${subject_agg_list[@]}"; do
                        for norm_type in "${norm_type_list[@]}"; do
                            for stft_type in "${stft_type_list[@]}"; do
                                # Launch experiment on the selected GPU
                                run_experiment $learning_rate $weight_decay $model_ver $downsampling_rate $mask_num $subject_agg $norm_type $stft_type $gpu_idx &

                                # Increment the job count for the current GPU
                                job_count=$((job_count + 1))

                                # If the number of jobs on this GPU reaches the limit, move to the next GPU
                                if [ $job_count -ge $jobs_per_gpu ]; then
                                    gpu_idx=$((gpu_idx + 1))
                                    job_count=0

                                    # If all GPUs are in use, wait for jobs to finish before reusing GPUs
                                    if [ $gpu_idx -ge ${#gpu_list[@]} ]; then
                                        gpu_idx=0
                                        wait  # Wait for all jobs to complete
                                    fi
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

wait  # Ensure all background jobs complete

# Print time spent & end date
end=$(date +%s)
echo "Experiment ended."
date
duration=$((end - start))
echo "Elapsed time: $(($duration / 3600)) hours, $(($duration / 60 % 60)) minutes, $(($duration % 60)) seconds"

exit 0
