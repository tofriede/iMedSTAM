#!/bin/bash

python train.py \
    -c configs/efficienttam_training/finetune_all_data.yaml \
    --use-cluster 0 \
    --num-gpus 8
