#!/bin/bash
#SBATCH --job-name=subwise
#SBATCH --output=slurm_logs/subwise_%A_%a.out
#SBATCH --error=slurm_logs/subwise_%A_%a.err #SBATCH --nodelist=node2,node4
#SBATCH --cpus-per-task=4 #fast : 16
#SBATCH --mem=10G 
#SBATCH --exclude=node1,node3,node4
#SBATCH --time=24:00:00
#SBATCH --array=0-50


conda activate CL_MRI_2

seed=$SLURM_ARRAY_TASK_ID

cd ..

for SUBJECT_NUM in {0..2}; do
    python main.py --seed $seed --subject_agg sub_wise --subject_num $SUBJECT_NUM
done



# Your script commands go here