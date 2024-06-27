module load pytorch/1.12
source /scratch/project_2007023/boris/envs/missing/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/imdb_input_test_${TIMESTAMP}.txt"


srun -N 1 -n 1 -c 3 -t 33:33:33 --mem-per-cpu=32G --gres=gpu:a100:2 \
     -p gpusmall --account=project_2002243 \
     python3 run.py with data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/mmimdb/ \
     num_gpus=2 \
     num_nodes=1 \
     per_gpu_batchsize=2 \
     task_finetune_mmimdb \
     input_prompts \
     step50k \
     load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/imdb_train_seed0_from_vilt_200k_mlm_itm/version_51/checkpoints/last.ckpt  \
     exp_name=imdb_input_test  \
     test_ratio=0.2 \
     test_type=full  \
     test_only=True \
     | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    END_TIME=$(date +%Y%m%d_%H%M%S)
    echo "Program finished at: $END_TIME" | tee -a "$LOG_FILE"

    START_SEC=$(date -d "$START_TIME" +%s)
    END_SEC=$(date -d "$END_TIME" +%s)
    DIFF_SEC=$((END_SEC - START_SEC))

    HOURS=$((DIFF_SEC / 3600))
    MINUTES=$(((DIFF_SEC % 3600) / 60))
    
    echo "Elapsed time: ${HOURS}h ${MINUTES}m" | tee -a "$LOG_FILE"
else
    echo "Program failed to complete." | tee -a "$LOG_FILE"
fi