MODEL_PATH=/data/hf-models/Qwen3-8B
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 推理测试
llamafactory-cli chat \
    --model_name_or_path /data/hf-models/Qwen3-8B \
    --template qwen \
    --infer_backend huggingface

llamafactory-cli chat \
    --model_name_or_path /home/zhangyue/data/pre-train-learning/output/qwen3_novel_full_pretrain \
    --adapter_name_or_path saves/qwen3_full_pt_sft4_mixed_lora \
    --template qwen \
    --infer_backend huggingface

llamafactory-cli chat \
    --model_name_or_path /home/zhangyue/data/pre-train-learning/output/qwen3_novel_full_pretrain \
    --adapter_name_or_path /home/zhangyue/data/LLaMA-Factory/saves/qwen3_full_pt_sft4_lora \
    --template qwen \
    --infer_backend huggingface


# 基于基础模型进行全参数微调
llamafactory-cli train qwen/qwen3_full_sft.yaml \
    --model_name_or_path /data/hf-models/Qwen3-8B \
    --output_dir saves/qwen3_full_sft \
    --dataset xd_final_sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --cutoff_len 512

# 基于预训练版进行sft
llamafactory-cli train qwen/qwen3_full_pt_sft.yaml 

# 基于预训练版进行sft,添加novel数据扩充
llamafactory-cli train qwen/qwen3_full_pt_sft2.yaml 


# 基于预训练版进行sft,添加novel数据扩充。增加训练轮次，降低学习率
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train qwen/qwen3_full_pt_sft3.yaml 

# 基于预训练版进行sft,添加novel数据扩充。增加训练轮次，降低学习率, 混合数据集
CUDA_VISIBLE_DEVICES=0,2,3,4 llamafactory-cli train qwen/qwen3_full_pt_sft4_mixed.yaml 

# 基于预训练版进行sft,添加novel数据扩充。增加训练轮次，降低学习率, 混合数据集, 添加lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 llamafactory-cli train qwen/qwen3_full_pt_sft4_mixed_lora.yaml 


# 基于预训练版进行sft,添加novel数据扩充。增加训练轮次，降低学习率, 混合数据集, 添加lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 llamafactory-cli train qwen/qwen3_full_pt_sft4_lora.yaml 




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