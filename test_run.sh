module load pytorch/1.12
source /scratch/project_2007023/boris/envs/missing/bin/activate

# 换测试的missing type的时候，需要修改===3===个地方的testm_
# with_delta_infer每次修改顺带需要修改===2===个with或without
# 切换prompt类型的时候,只需要ctrl F寻找到该类型替换即可

srun -N 1 -n 1 -c 3 -t 00:15:00 --mem-per-cpu=32G --gres=gpu:a100:2 \
        -p gputest --account=project_2002243 \
        python3 run.py with \
        data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/hateful_memes/ \
        task_finetune_hateful_memes \
        kronecker_prompts \
        trainm_t_testm_t \
        load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/hateful_memes_kronecker_trainm_t_testm_t_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/last.ckpt \
        with_delta_infer=True \
        test_exp_name=test_only \
        test_only=True \
        | tee -a /scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/test_only.txt

