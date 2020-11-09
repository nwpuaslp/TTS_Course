#!/usr/bin/env bash

work_path=$(dirname $(dirname $(dirname $(readlink -f $0))))
checkpoint=${work_path}/log_dir/example/Tacotron/Tacotron.ckpt-10000
labels=${work_path}/testdata/biaobei/labels
output_dir=${work_path}/model_outputs/example/synthesized

python3 ${work_path}/infer.py \
   --yaml_conf=${work_path}/hparams.yaml \
   --checkpoint=${checkpoint} \
   --label_dir=${labels} \
   --output_dir=${output_dir} \
   --hparams="outputs_per_step=2"
