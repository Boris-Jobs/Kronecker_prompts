module load pytorch/1.12
source /scratch/project_2007023/boris/envs/missing/bin/activate


srun -N 1 -n 1 -c 3 -t 20:15:00 --mem-per-cpu=32G --gres=gpu:a100:2 \
     -p gpusmall --account=project_2002243 \
     python3 run.py with \
     data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/mmimdb/ \
     task_finetune_mmimdb \
     input_prompts \
     trainm_t_testm_t \
     load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/vilt/vilt_200k_mlm_itm.ckpt \
     exp_name=mmimdb_input_trainm_t_testm_t \
     | tee -a /scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/mmimdb_input_trainm_t_testm_t.txt