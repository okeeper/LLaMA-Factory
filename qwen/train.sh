MODEL_PATH=/data/hf-models/Qwen3-8B

# 全参数微调
llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --dataset xd_final_sft,xd_pretain_mix \
    --template qwen3 \
    --cutoff_len 2048 \
    --max_samples 10000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/qwen3_full_sft \
    --logging_steps 1 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000