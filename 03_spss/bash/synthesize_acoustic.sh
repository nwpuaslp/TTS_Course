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
mgc_dir="${output_dir}/mgc"
bap_dir="${output_dir}/bap"
lf0_dir="${output_dir}/lf0"
cmp_dir="${output_dir}/cmp"
syn_dir="${output_dir}/syn_dir"
syn_wav_dir="${output_dir}/syn_wav"

mkdir -p ${output_dir}
mkdir -p ${mgc_dir}
mkdir -p ${bap_dir}
mkdir -p ${lf0_dir}
mkdir -p ${cmp_dir}
mkdir -p ${syn_dir}
mkdir -p ${syn_wav_dir}

python synthesize.py --checkpoint=$checkpoint --label_dir=$label_dir --output_dir=$output_path --model_type=AcousticModel


echo "$0 $@"  #
# tools directory
world="tools/bin/World_v2"
sptk="tools/bin/SPTK-3.9"

# initializations
fs=16000

if [ "$fs" -eq 16000 ]
then
nFFTHalf=1024
alpha=0.58
fi

if [ "$fs" -eq 48000 ]
then
nFFTHalf=2048
alpha=0.77
fi

mcsize=59
order=4

echo ${mgc_dir}

for file in $mgc_dir/*.mgc #.mgc
do
    filename="${file##*/}"
    file_id="${filename%.*}"

    echo $file_id

    ### WORLD Re-synthesis -- reconstruction of parameters ###

    ### convert lf0 to f0 ###
    $sptk/sopr -magic -1.0E+10 -EXP -MAGIC 0.0 ${lf0_dir}/$file_id.lf0 | $sptk/x2x +fa > ${syn_dir}/$file_id.syn.f0a
    $sptk/x2x +ad ${syn_dir}/$file_id.syn.f0a > ${syn_dir}/$file_id.syn.f0

    ### convert mgc to sp ###
    $sptk/mgc2sp -a $alpha -g 0 -m $mcsize -l $nFFTHalf -o 2 ${mgc_dir}/$file_id.mgc | $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > ${syn_dir}/$file_id.syn.sp

    ### convert bap to ap ###
    $sptk/mgc2sp -a $alpha -g 0 -m $order -l $nFFTHalf -o 2 ${bap_dir}/$file_id.bap | $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > ${syn_dir}/$file_id.syn.ap

    $world/synthesis ${syn_dir}/$file_id.syn.f0 ${syn_dir}/$file_id.syn.sp ${syn_dir}/$file_id.syn.ap ${syn_wav_dir}/$file_id.syn.wav
done

rm -rf $syn_dir
