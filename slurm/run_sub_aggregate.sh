#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --output=slurm_logs/aggregate_%A_%a.out
#SBATCH --error=slurm_logs/aggregate_%A_%a.err #SBATCH --nodelist=node2,node4
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node1,node3,node4
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --array=0-50


conda activate CL_MRI_2

seed=$SLURM_ARRAY_TASK_ID

cd ..

python main.py --seed $seed --subject_agg sub_agg



# Your script commands go here