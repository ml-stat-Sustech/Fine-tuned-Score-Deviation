#!/bin/bash

# ! check the train_size
# ! check the seed
export CUDA_VISIBLE_DEVICES=1,0

accelerate launch sft.py \
        --seed 42 \
        --gradient_checkpointing \
        --fp16 \
        --ckpts_dir ./ckpts/model \
        --batch_size 8 \
        --num_workers 8 \
        --learning_rate 1e-3 \
        --num_train_epochs 3 \
        --train_size 0.3 \
        --split train \
        --num_warmup_steps 10 \
        