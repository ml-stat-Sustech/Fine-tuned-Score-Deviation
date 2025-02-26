#!/bin/bash

# ! check the train_size
# ! check the seed

python eval.py \
        --dataset WikiMIA \
        --split test \
        --train_size 0.3 \
        --model_path huggyllama/llama-7b \
        --fine_tuned_para ckpts/model/llama-7b/seed_42/WikiMIA \
        --seed 42 