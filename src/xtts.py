import os
import json
import argparse

from utils import formatter
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.manage import ModelManager
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig,
)


def get_arg_parser():
    parser = argparse.ArgumentParser(description="XTTS Training Script")

    # Logging parameters
    parser.add_argument("--run_name", type=str, default="GPT_XTTS_MUNDARI_FT")
    parser.add_argument("--wandb_project", type=str, default="mundari-tts")
    parser.add_argument("--dashboard_logger", type=str, default="wandb")
    parser.add_argument("--wandb_entity", type=str, default="coqui")
    parser.add_argument(
        "--language", type=str, default="hi"
    )  # language to be used to tokenize the text
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="mundari-tts")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--optimizer_wd_only_on_weights", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_group_size", default=48, type=int)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--in_sampling_rate", type=int, default=22050)
    parser.add_argument("--out_sampling_rate", type=int, default=24000)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--num_workers_eval", default=8, type=int)
    parser.add_argument("--use_noise_augment", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad_clip", type=float, default=1000.0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr_scheduler", type=str, default="MultiStepLR")
    parser.add_argument(
        "--optimizer_params",
        type=json.loads,
        default='{"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}',
    )
    parser.add_argument(
        "--lr_scheduler_params",
        type=json.loads,
        default='{"milestones": [900000, 2700000, 5400000], "gamma": 0.5, "last_epoch": -1}',
    )

    parser.add_argument("--max_text_length", type=int, default=400)
    parser.add_argument("--max_wav_length", type=int, default=441000)
    parser.add_argument("--max_conditioning_length", type=int, default=132300)
    parser.add_argument("--min_conditioning_length", type=int, default=66150)

    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--use_speaker_weighted_sampler", action="store_true")
    parser.add_argument("--speaker_weighted_sampler_alpha", default=1.0, type=float)
    parser.add_argument("--add_blank", action="store_true")
    parser.add_argument("--shuffle", action="store_true")

    # DVAE files
    parser.add_argument(
        "--dvae_checkpoint_link",
        type=str,
        default="https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth",
    )
    parser.add_argument(
        "--mel_norm_link",
        type=str,
        default="https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth",
    )
    parser.add_argument(
        "--tokenizer_file_link",
        type=str,
        default="https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
    )
    parser.add_argument(
        "--xtts_checkpoint_link",
        type=str,
        default="https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth",
    )
    return parser


def main(args):
    CHECKPOINTS_OUT_PATH = os.path.join(args.output_path, "pretrained_checkpoint")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(
        CHECKPOINTS_OUT_PATH, os.path.basename(args.dvae_checkpoint_link)
    )
    MEL_NORM_FILE = os.path.join(
        CHECKPOINTS_OUT_PATH, os.path.basename(args.mel_norm_link)
    )

    TOKENIZER_FILE = os.path.join(
        CHECKPOINTS_OUT_PATH, os.path.basename(args.tokenizer_file_link)
    )
    XTTS_CHECKPOINT = os.path.join(
        CHECKPOINTS_OUT_PATH, os.path.basename(args.xtts_checkpoint_link)
    )

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" | > Downloading DVAE files!\n")
        ModelManager._download_model_files(
            [args.mel_norm_link, args.dvae_checkpoint_link],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True,
        )

    # download XTTS files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(" | > Downloading XTTS v2 files!\n")
        ModelManager._download_model_files(
            [args.tokenizer_file_link, args.xtts_checkpoint_link],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True,
        )

    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        args.dataset_name,
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_val.csv",
        path=args.dataset_path,
        language=args.language,
    )

    # init args and config
    model_args = model_args = {
        "max_conditioning_length": args.max_conditioning_length,
        "min_conditioning_length": args.min_conditioning_length,
        "max_wav_length": args.max_wav_length,
        "max_text_length": args.max_text_length,
        "mel_norm_file": MEL_NORM_FILE,
        "dvae_checkpoint": DVAE_CHECKPOINT,
        "xtts_checkpoint": XTTS_CHECKPOINT,
        "tokenizer_file": TOKENIZER_FILE,
        "gpt_num_audio_tokens": 1026,
        "gpt_start_audio_token": 1024,
        "gpt_stop_audio_token": 1025,
    }
    # define audio config
    audio_config = {
        "sample_rate": args.in_sampling_rate,
        "dvae_sample_rate": args.in_sampling_rate,
        "output_sample_rate": args.out_sampling_rate,
    }
    # training parameters config
    training_config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "audio": XttsAudioConfig(**audio_config),
        "output_path": args.output_path,
        "model_args": GPTArgs(**model_args),
        "run_name": args.run_name,
        "project_name": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "dashboard_logger": args.dashboard_logger,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "batch_group_size": args.batch_group_size,
        "num_loader_workers": args.num_workers,
        "num_eval_loader_workers": args.num_workers_eval,
        "print_step": 100,
        "plot_step": 100,
        "log_model_step": 1000,
        "save_step": 10000,
        "save_n_checkpoints": 1,
        "save_checkpoints": True,
        "print_eval": False,
        "run_eval": True,
        "test_sentences": [],
        "shuffle": args.shuffle,
        "add_blank": args.add_blank,
        "mixed_precision": args.mixed_precision,
        "use_noise_augment": args.use_noise_augment,
        "optimizer_wd_only_on_weights": args.optimizer_wd_only_on_weights,
        "use_speaker_weighted_sampler": args.use_speaker_weighted_sampler,
        "speaker_weighted_sampler_alpha": args.speaker_weighted_sampler_alpha,
        "optimizer": args.optimizer,
        "lr_scheduler": args.lr_scheduler,
        "optimizer_params": args.optimizer_params,
        "lr_scheduler_params": args.lr_scheduler_params,
    }

    # init the model from config
    model = GPTTrainer.init_from_config(**training_config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config_dataset, eval_split=True, formatter=formatter
    )

    print(f" | > Train Samples: {len(train_samples)}")
    print(f" | > Eval Samples: {len(eval_samples)}")

    training_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=False,
        grad_accum_steps=args.grad_accum_steps,
    )

    trainer = Trainer(
        training_args,
        training_config,
        model=model,
        output_path=args.output_path,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
