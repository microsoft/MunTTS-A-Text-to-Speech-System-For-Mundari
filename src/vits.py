import os
import json
import argparse
from utils import formatter
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import (
    BaseAudioConfig,
    BaseDatasetConfig,
    CharactersConfig,
)
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Training and evaluation script for acoustic / e2e TTS model "
    )

    # dataset parameters
    parser.add_argument("--dataset_name", default="mundari-tts", type=str)
    parser.add_argument("--language", default="unr", type=str)
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_phonemes", action="store_true")
    parser.add_argument("--phoneme_language", default="en-us", choices=["en-us"])
    parser.add_argument("--add_blank", action="store_true")
    parser.add_argument(
        "--text_cleaner",
        default="multilingual_cleaners",
        choices=["multilingual_cleaners"],
    )
    parser.add_argument("--sampling_rate", default=44100, type=int)
    parser.add_argument("--resample_to_sampling_rate", action="store_true")
    parser.add_argument("--do_trim_silence", action="store_true")
    parser.add_argument("--eval_split_size", default=0.01)
    parser.add_argument("--min_audio_len", default=1)
    parser.add_argument("--max_audio_len", default=float("inf"))  # 20*22050
    parser.add_argument("--min_text_len", default=1)
    parser.add_argument("--max_text_len", default=float("inf"))  # 400
    parser.add_argument("--signal_norm", action="store_true")

    # logging
    parser.add_argument("--dashboard_logger", default="wandb", type=str)
    parser.add_argument("--wandb_entity", default="coqui", type=str)
    parser.add_argument("--wandb_project", default="mundari-tts", type=str)

    # model parameters
    parser.add_argument(
        "--model",
        default="vits",
        choices=["vits"],
    )
    parser.add_argument("--use_speaker_embedding", action="store_true")
    parser.add_argument("--use_speaker_weighted_sampler", action="store_true")
    parser.add_argument("--speaker_weighted_sampler_alpha", default=1.0, type=float)
    # training parameters
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--batch_size_eval", default=32, type=int)
    parser.add_argument("--batch_group_size", default=0, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--num_workers_eval", default=8, type=int)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--use_noise_augment", action="store_true")

    # training - logging parameters
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_delay_epochs", default=0, type=int)
    parser.add_argument("--run_eval_steps", default=None, type=int)
    parser.add_argument("--print_step", default=100, type=int)
    parser.add_argument("--plot_step", default=100, type=int)
    parser.add_argument("--save_step", default=10000, type=int)
    parser.add_argument("--save_n_checkpoints", default=1, type=int)
    parser.add_argument("--save_best_after", default=10000, type=int)

    # distributed training parameters
    parser.add_argument("--continue_path", default="", type=str)
    parser.add_argument("--restore_path", default="", type=str)
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--group_id", default="", type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--port", default=54321, type=int)

    # vits
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--use_sdp", action="store_true")
    parser.add_argument("--lr_disc", default=0.0002, type=float)  # for vits
    parser.add_argument("--lr_gen", default=0.0002, type=float)  # for vits
    parser.add_argument(
        "--lr_scheduler_gen", default="ExponentialLR", type=str
    )  # for vits
    parser.add_argument(
        "--lr_scheduler_disc", default="ExponentialLR", type=str
    )  # for vits
    parser.add_argument(
        "--lr_scheduler_gen_params",
        default='{"gamma": 0.999875, "last_epoch": -1}',
        type=json.loads,
    )  # for vits
    parser.add_argument(
        "--lr_scheduler_disc_params",
        default='{"gamma": 0.999875, "last_epoch": -1}',
        type=json.loads,
    )  # for vits
    return parser


def main(args):
    # set dataset config
    dataset_config = BaseDatasetConfig(
        args.dataset_name,
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_val.csv",
        path=args.dataset_path,
        language=args.language,
    )

    samples, _ = load_tts_samples(dataset_config, eval_split=False, formatter=formatter)
    texts = "".join(item["text"] for item in samples)
    lang_chars = list(set(texts)) + [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    lang_chars = sorted(set(lang_chars))
    del samples, texts

    # set characters config
    characters_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="".join(lang_chars),
        punctuations="+<=>!¡'(),-.:;¿? ",
        phonemes=None,
    )

    # set base tts config
    base_tts_config = {
        "audio": BaseAudioConfig(
            trim_db=60.0,
            mel_fmin=0.0,
            mel_fmax=8000,
            log_func="np.log",
            spec_gain=1.0,
            signal_norm=args.signal_norm,
            sample_rate=args.sampling_rate,
            resample=args.resample_to_sampling_rate,
        ),
        "use_phonemes": args.use_phonemes,
        "phoneme_language": args.phoneme_language,
        "characters": characters_config,
        "add_blank": args.add_blank,
        "text_cleaner": args.text_cleaner,
        "use_noise_augment": args.use_noise_augment,
        # dataset
        "datasets": [dataset_config],
        "min_audio_len": args.min_audio_len,
        "max_audio_len": args.max_audio_len,
        "min_text_len": args.min_text_len,
        "max_text_len": args.max_text_len,
        # data loading
        "num_loader_workers": args.num_workers,
        "num_eval_loader_workers": args.num_workers_eval,
        # model
        "use_d_vector_file": False,
        # trainer - run
        "shuffle": args.shuffle,
        "output_path": args.output_path,
        # wandb logging
        "dashboard_logger": args.dashboard_logger,
        "project_name": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "run_name": f"{args.language}_{args.model}_all_{args.sampling_rate}",
        # trainer - logging
        "run_eval_steps": args.run_eval_steps,
        "print_step": args.print_step,
        "plot_step": args.plot_step,
        # trainer - checkpointing
        "save_step": args.save_step,
        "save_n_checkpoints": args.save_n_checkpoints,
        "save_best_after": args.save_best_after,
        # trainer - eval
        "print_eval": False,
        "run_eval": True,
        # trainer - test
        "test_delay_epochs": args.test_delay_epochs,
        # trainer - training
        "distributed_url": f"tcp://localhost:{args.port}",
        "mixed_precision": args.mixed_precision,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.batch_size_eval,
        "batch_group_size": args.batch_group_size,
        "lr": args.lr,
        "lr_disc": args.lr_disc,
        "lr_gen": args.lr_gen,
        "lr_scheduler_gen": args.lr_scheduler_gen,
        "lr_scheduler_disc": args.lr_scheduler_disc,
        "lr_scheduler_gen_params": args.lr_scheduler_gen_params,
        "lr_scheduler_disc_params": args.lr_scheduler_disc_params,
        # test
        "test_sentences": [],
        "eval_split_size": args.eval_split_size,
        # speaker
        "use_speaker_embedding": args.use_speaker_embedding,
        "use_speaker_weighted_sampler": args.use_speaker_weighted_sampler,
        "speaker_weighted_sampler_alpha": args.speaker_weighted_sampler_alpha,
    }

    vitsArgs = VitsArgs(
        num_speakers=2,  # male and female
        use_sdp=args.use_sdp,
        use_speaker_embedding=args.use_speaker_embedding,
    )
    config = VitsConfig(
        **base_tts_config,
        model_args=vitsArgs,
    )

    # set preprocessors
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # load data
    train_samples, eval_samples = load_tts_samples(
        dataset_config, eval_split=True, formatter=formatter
    )
    print(f" | > Train Samples: {len(train_samples)}")
    print(f" | > Eval Samples: {len(eval_samples)}")

    # set speaker manager
    if args.use_speaker_embedding:
        speaker_manager = SpeakerManager()
        speaker_manager.set_ids_from_data(
            train_samples + eval_samples, parse_key="speaker_name"
        )
    else:
        speaker_manager = None

    model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

    training_args = TrainerArgs(
        continue_path=args.continue_path,
        restore_path=args.restore_path,
        rank=args.rank,
        group_id=args.group_id,
        use_ddp=args.use_ddp,
    )

    # set trainer
    trainer = Trainer(
        training_args,
        config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # run training
    trainer.fit()


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
