MODEL_PATH=/data/hf-models/Qwen3-8B

llamafactory-cli train qwen/qwen3_full_sft.yaml

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3

# 全参数微调
llamafactory-cli train \
    --model_name_or_path /data/hf-models/Qwen3-8B \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --dataset xd_final_sft \
    --template qwen \
    --cutoff_len 512 \
    --max_samples 10000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/qwen3_full_sft \
    --logging_steps 1 \
    --save_steps 500 \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16