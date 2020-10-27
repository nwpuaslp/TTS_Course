#!/usr/bin/env bash
if [ "$#" != "1" ]; then
    echo "Usage: $0 <valid_size>"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=''
valid_size=$1
work_path=$(dirname $(dirname $(dirname $(readlink -f $0))))
data_root=${work_path}/testdata/biaobei

find ${data_root}/labels -name '*.lab' |shuf  > ${data_root}/file.scp
awk -v count=${valid_size} '{if (NR > count) { print $0 > "'${data_root}/train_scp'"; } else { print $0 > "'${data_root}/valid_scp'"; }}' ${data_root}/file.scp
rm ${data_root}/file.scp

python3 ${work_path}/preprocess.py \
    --yaml_conf=${work_path}/hparams.yaml \
    --label_dir=${data_root}/labels \
    --wav_dir=${data_root}/wavs \
    --out_feature_dir=${data_root}/acoustic_features \
