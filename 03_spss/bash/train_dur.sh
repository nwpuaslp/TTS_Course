#!/usr/bin/env bash

# Example: bash train.sh data/labels data/mels
# export CUDA_VISIBLE_DEVICES=''

current_working_dir=$(pwd)
label_dir=train_data/duration_features
mel_dir=train_data/duration_targets

cd $label_dir

ls *.npy  | awk -F'.' '{print $1}'  >  $current_working_dir/train.scp

cd $current_working_dir

python train.py --acoustic_features_dir=$mel_dir --labels_dir=$label_dir --filelist=train.scp --log_dir=logdir_dur --model_type=DurationModel
