module load pytorch/1.12
source /scratch/project_2003238/v/envs/missing/bin/activate

srun -N 1 -n 1 -c 3 -t 00:15:00 --mem-per-cpu=32G --gres=gpu:a100:2 \
     -p gputest --account=project_2003238 \
     python3 run.py with \
     data_root=/scratch/project_2003238/v/missing_aware_prompts/Kronecker_prompts/datasets/hateful_memes/ \
     task_finetune_hateful_memes \
     kronecker_prompts \
     trainm_t_testm_t \
     load_path=/scratch/project_2003238/v/missing_aware_prompts/Kronecker_prompts/vilt/vilt_200k_mlm_itm.ckpt \
     exp_name=EDL_loss \

# | tee -a /scratch/project_2003238/v/missing_aware_prompts/missing_aware_prompts/result/hateful_memes_kronecker_trainm_t_testm_t.txt