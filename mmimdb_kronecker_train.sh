module load pytorch/1.12
source /scratch/project_2007023/boris/envs/missing/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/${TIMESTAMP}_mmimdb_kronecker_train.txt"


echo "Starting training at: $(date)" | tee -a "$LOG_FILE"
srun -N 1 -n 1 -c 3 -t 33:33:33 --mem-per-cpu=32G --gres=gpu:a100:2 \
     -p gpusmall --account=project_2002243 \
     python3 run.py with \
     data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/mmimdb/ \
     num_gpus=2 \
     num_nodes=1 \
     task_finetune_mmimdb \
     kronecker_prompts \
     step50k \
     load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/vilt/vilt_200k_mlm_itm.ckpt \
     exp_name=${TIMESTAMP}_mmimdb_kronecker_train \
     | tee -a "$LOG_FILE"


if [ $? -eq 0 ]; then
    echo "Training completed successfully. Now to infer with delta." | tee -a "$LOG_FILE"
    
    srun -N 1 -n 1 -c 3 -t 01:00:00 --mem-per-cpu=32G --gres=gpu:a100:2 \
         -p gpusmall --account=project_2002243 \
         python3 run.py with \
         data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/mmimdb/ \
         num_gpus=2 \
         num_nodes=1 \
         task_finetune_mmimdb \
         kronecker_prompts \
         step50k \
         load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/${TIMESTAMP}_mmimdb_kronecker_train_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/last.ckpt \
         with_delta_infer=True \
         exp_name=${TIMESTAMP}_mmimdb_kronecker_test_with_delta \
         test_only=True \
         | tee -a "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "Testing with delta completed successfully. Now to infer without delta." | tee -a "$LOG_FILE"
        
        srun -N 1 -n 1 -c 3 -t 01:00:00 --mem-per-cpu=32G --gres=gpu:a100:2 \
             -p gpusmall --account=project_2002243 \
             python3 run.py with \
             data_root=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/datasets/mmimdb/ \
             num_gpus=2 \
             num_nodes=1 \
             task_finetune_mmimdb \
             kronecker_prompts \
             step50k \
             load_path=/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/${TIMESTAMP}_mmimdb_kronecker_train_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/last.ckpt \
             with_delta_infer=False \
             exp_name=${TIMESTAMP}_mmimdb_kronecker_test_without_delta \
             test_only=True \
             | tee -a "$LOG_FILE"
             
        if [ $? -eq 0 ]; then
            echo "Testing without delta completed successfully at: $(date)" | tee -a "$LOG_FILE"
        else
            echo "Testing without delta failed at: $(date)" | tee -a "$LOG_FILE"
        fi
    else
        echo "Testing with delta failed at: $(date)" | tee -a "$LOG_FILE"
    fi
else
    echo "Training failed. Aborting inference." | tee -a "$LOG_FILE"
fi
