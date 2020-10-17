#!/usr/bin/env bash

# Example: bash synthesis.sh label_dir output model.ckpt-10000
export CUDA_VISIBLE_DEVICES=''  
if [ "$#" != "3" ]; then
    echo "Usage: $0 <label_dir> <output_path> <checkpoint>"
    exit 1
fi

label_dir=$1
output_path=$2
checkpoint=$3

output_dir=${output_path}
mkdir -p ${output_dir}

python synthesize.py --checkpoint=$checkpoint --label_dir=$label_dir --output_dir=$output_path --model_type=DurationModel

