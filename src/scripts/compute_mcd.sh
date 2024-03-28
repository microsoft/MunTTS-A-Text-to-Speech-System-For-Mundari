#!/bin/bash

gen_dir=$1
gt_dir=$2
sampling_rate=$3

for gender in male female all; do
    python compute_mcd.py \
        --gen_wavdir $gen_dir \
        --gt_wavdir $gt_dir \
        --metadata_file $gt_dir/metadata_test.csv \
        --speaker $gender \
        --sampling_rate $sampling_rate
done