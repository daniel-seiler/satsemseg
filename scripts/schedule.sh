#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=satsemseg

python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
