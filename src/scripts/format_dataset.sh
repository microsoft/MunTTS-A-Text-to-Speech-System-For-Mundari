src_dir=$1
dest_dir=$2
sampling_rate=$3

rm -rf $dest_dir

python format_dataset.py \
    --source $src_dir \
    --destination $dest_dir \
    --devtest_split 0.1 \
    --random_state 42 \
    --restructure_directory \
    --normalize_text \
    --sampling_rate $sampling_rate \
