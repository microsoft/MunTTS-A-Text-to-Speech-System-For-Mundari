#!/bin/bash

infname=$1
out_dir=$2

mkdir -p $out_dir

python test_mms.py \
    --model "facebook/mms-tts-unr" \
    --text_file $infname \
    --output_dir $out_dir 

python audio_enhance.py $out_dir ${out_dir}_cleaned