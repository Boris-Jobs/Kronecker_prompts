module load pytorch/1.12
source /scratch/project_2007023/boris/envs/missing/bin/activate


srun -N 1 -n 1 -c 3 -t 33:33:33 --mem-per-cpu=32G --gres=gpu:a100:1 \
     -p gpusmall --account=project_2002243 \
     python3 run.py with \
     data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/hateful_memes/ \
     task_finetune_hateful_memes \
     kronecker_prompts \
     trainm_b_testm_b \
     load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/vilt/vilt_200k_mlm_itm.ckpt \
     exp_name=perfect_hateful_memes_kronecker_trainm_b \
     | tee -a /scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/perfect_hateful_memes_kronecker_trainm_b.txt


