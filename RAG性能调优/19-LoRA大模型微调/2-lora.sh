#!/bin/bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen3-0.6B \
    --train_type lora \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' 'swift/self-cognition:empty_think#600' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --loss_scale ignore_empty_think \
    --model_author swift \
    --model_name swift-robot



CUDA_VISIBLE_DEVICES=0  swift infer  --adapters output/v0-20250725-005426/checkpoint-100 --stream true  --temperature 0  --max_new_tokens 512


swift export --adapters output/v0-20250725-005426/checkpoint-100 --merge_lora true 

CUDA_VISIBLE_DEVICES=0 swift infer --model output/v0-20250725-005426/checkpoint-100-merged 