#!/bin/bash

set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=Qwen/Qwen3.5-9B

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset dataset_stage_1 \
    --template qwen3_5 \
    --cutoff_len 40960 \
    --max_new_tokens 32768 \
    --max_samples 10000 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/qwen3.5-9b/lora/sft/dataset_stage_1 \
    --logging_steps 5 \
    --save_steps 68 \
    --plot_loss \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 1800 \
    --val_size 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 34 \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --flash_attn fa2 \
    --enable_liger_kernel 