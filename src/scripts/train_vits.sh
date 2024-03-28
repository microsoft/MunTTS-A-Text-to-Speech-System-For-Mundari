in_dir=$1
out_dir=$2
sampling_rate=${3:-"44100"}

CUDA_VISIBLE_DEVICES='0' python vits.py \
    --dataset_path $in_dir \
    --output_path $out_dir \
    --dataset_name mundari-tts \
    --sampling_rate $sampling_rate \
    --language unr \
    --model vits \
    --lr_gen 5e-4 \
    --lr_disc 5e-4 \
    --use_speaker_embedding \
    --use_speaker_weighted_sampler \
    --batch_size 128 \
    --batch_size_eval 128 \
    --epochs 2500 \
    --mixed_precision \
    --do_trim_silence \
    --add_blank \
    --use_sdp \
    --shuffle