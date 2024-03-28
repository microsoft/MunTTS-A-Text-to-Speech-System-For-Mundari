in_dir=$1
out_dir=$2

CUDA_VISIBLE_DEVICES='0' python xtts.py \
    --output_path $out_dir \
    --dataset_path $in_dir \
    --language hi \
    --dataset_name mundari-tts \
    --batch_size 16 \
    --eval_batch_size 16 \
    --grad_accum_steps 16 \
    --shuffle \
    --add_blank \
    --mixed_precision \
    --optimizer_wd_only_on_weights \
    --use_speaker_weighted_sampler