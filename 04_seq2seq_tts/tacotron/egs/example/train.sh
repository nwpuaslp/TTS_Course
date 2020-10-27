#!/usr/bin/env bash

work_path=$(dirname $(dirname $(dirname $(readlink -f $0))))
dataset_dir=${work_path}/testdata/biaobei/
log_dir=${work_path}/log_dir/example

python3 -u ${work_path}/train.py \
   --yaml_conf=${work_path}/hparams.yaml \
   --log_dir=${log_dir} \
   --train_filelist=${dataset_dir}/train_scp \
   --valid_filelist=${dataset_dir}/valid_scp \
   --acoustic_features_dir=${dataset_dir}/acoustic_features \
   --labels_dir=${dataset_dir}/labels \
   --hparams="batch_size=64,
              outputs_per_step=2"
