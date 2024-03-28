#!/bin/bash

in_dir=$1
out_dir=$2

rm -rf $out_dir
mkdir -p $out_dir

for wav_name in `ls ${in_dir}`; do
    ffmpeg -y -loglevel quiet -i ${in_dir}/${wav_name} -af "highpass=500,lowpass=8000,afftdn,silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-60dB" ${out_dir}/${wav_name}
done 