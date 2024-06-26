module load pytorch/1.12
source /scratch/project_2007023/boris/envs/missing/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/imdb_train_${TIMESTAMP}.txt"

# srun -N 1 -n 1 -t 00:15:00 --mem-per-cpu=32G --gres=gpu:a100:2 \
#      -p gputest --account=project_2002243 python3 checkpoint_check.py \
#      | tee "$LOG_FILE"

srun -N 1 -n 1 -c 3 -t 00:30:00 --mem-per-cpu=32G --gres=gpu:a100:2 \
     -p gpusmall --account=project_2002243 \
     python3 run.py with \
     data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/mmimdb/ \
     num_gpus=2 \
     num_nodes=1 \
     per_gpu_batchsize=2 \
     task_finetune_mmimdb \
     load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/vilt/vilt_200k_mlm_itm.ckpt \
     exp_name=imdb_train \
     | tee -a "$LOG_FILE"
