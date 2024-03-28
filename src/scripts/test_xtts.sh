#!/bin/bash

ckpt_dir=$1
data_dir=$2
male_speaker_ref_fname=$3
female_speaker_ref_fname=$4

gen_dir="${ckpt_dir}/wavs"
rm -rf $gen_dir
mkdir -p $gen_dir

test_infname="${data_dir}/metadata_test.csv"

while IFS=$' ' read -r line || [[ -n "$line" ]]; do
    IFS='$' read -ra parts <<< "$line"
    
    user_id="${parts[0]}"
    text="${parts[1]}"
    gender="${parts[2]}"

    text=$(python -c "print('$text'.replace(' , ', ', ').replace(' । ', '। ').strip())")

    # select speaker based on gender 
    if [ "$gender" == "male" ]; then
        speaker_ref_fname=$male_speaker_ref_fname
    else
        speaker_ref_fname=$female_speaker_ref_fname
    fi
    
    tts --text "$text" \
        --model_path "$ckpt_dir" \
        --config_path "${ckpt_dir}/config.json" \
        --out_path "${gen_dir}/test_${user_id}.wav" \
        --speaker_wav "${data_dir}/wavs/${speaker_ref_fname}" \
        --language_idx hi \
        --use_cuda t \
        --device cuda

done < "$test_infname"

python scripts/audio_enhance.py $gen_dir ${gen_dir}_cleaned