#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=''

current_working_dir=$(pwd)
label_dir=train_data/acoustic_features

cd $label_dir

ls *.npy  | awk -F'.' '{print $1}'  >  $current_working_dir/train.scp

cd $current_working_dir

python prepare.py