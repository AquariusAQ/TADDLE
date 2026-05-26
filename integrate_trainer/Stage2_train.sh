#!/bin/bash

set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=Qwen/Qwen3.5-9B
LoRA_PATH=./saves/qwen3.5-9b/lora/sft/dataset_stage_1

CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${LoRA_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset dataset_stage_2 \
    --template qwen3_5 \
    --cutoff_len 40960 \
    --max_new_tokens 32768 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/qwen3.5-9b/lora/sft/dataset_stage_2 \
    --logging_steps 5 \
    --save_steps 133 \
    --plot_loss \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-6 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 \
    --ddp_timeout 1800 \
    --val_size 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 33 \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --flash_attn fa2 \
    --enable_liger_kernel
