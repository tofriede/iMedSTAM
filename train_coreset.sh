#!/bin/bash

python train.py \
    -c configs/efficienttam_training/finetune_coreset.yaml \
    --use-cluster 0 \
    --num-gpus 4
