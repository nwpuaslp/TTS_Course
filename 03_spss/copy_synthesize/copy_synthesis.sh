#!/usr/bin/env bash

if [ "$#" != "1" ]; then
     echo "Usage: $0 <wav> "
     exit 1
fi

wav_path=$1
work_dir=$(pwd)

world="${work_dir}/tools/bin/World"

$world/analysis ${wav_path} ${world}/synthesis.f0 ${world}/synthesis.sp ${world}/synthesis.ap

$world/synthesis ${world}/synthesis.f0 ${world}/synthesis.sp ${world}/synthesis.ap ${work_dir}/copy_synthesize/syn.wav
