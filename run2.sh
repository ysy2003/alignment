#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# python main.py
# PYTHONPATH=. python experiment/train.py
PYTHONPATH=. python experiment/evaluate_no_retrieve.py