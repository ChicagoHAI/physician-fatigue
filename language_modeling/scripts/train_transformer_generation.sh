#!/bin/bash -l
#SBATCH --job-name=lm    # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=256gb                    # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=32 # 
#SBATCH --output=log/training.log
#SBATCH --error=log/training.err
#SBATCH --partition=Long
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=SIGUSR1@120

source ~/anaconda3/etc/profile.d/conda.sh
conda activate md_lm
cd ./language_modeling

# srun 
TRANSFORMERS_OFFLINE=1 python -m models.note_generation.main \
    --task lm \
    --note_type comments \
    --num_doctors 51 \
    --max_epochs 5 \
    --max_seq_length 1024 \
    --output_dir ./data/lm/exp \
    --data_dir./data/lm/ \
    --model_name_or_path gpt2 \
    --warmup_steps 500 \
    --gpus 4 \
    --do_train \
    --do_predict \
    --cache_dir ./data/transformers \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 5e-5 \
    --overwrite_dir \
    --fp16 
    # --gradient_accumulation_steps 8 \