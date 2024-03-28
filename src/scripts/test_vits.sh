#!/bin/bash

ckpt_dir=$1
data_dir=$2

gen_dir="${ckpt_dir}/wavs"
rm -rf $gen_dir
mkdir -p $gen_dir

test_infname="${data_dir}/metadata_test.csv"

while IFS=$' ' read -r line || [[ -n "$line" ]]; do
    IFS='$' read -ra parts <<< "$line"
    
    user_id="${parts[0]}"
    text="${parts[1]}"
    gender="${parts[2]}"
    
    tts --text "$text" \
        --model_path "${ckpt_dir}/best_model.pth" \
        --config_path "${ckpt_dir}/config.json" \
        --out_path "${gen_dir}/test_${user_id}.wav" \
        --speaker_idx "$gender" \
        --use_cuda t \
        --device cuda

done < "$test_infname"

python scripts/audio_enhance.py $gen_dir ${gen_dir}_cleaned