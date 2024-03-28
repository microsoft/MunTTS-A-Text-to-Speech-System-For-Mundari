import os
import re
import argparse
import librosa as lr
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from shutil import copyfile, rmtree
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, help="Path to the source directory", required=True
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="Path to the destination directory",
        required=True,
    )

    parser.add_argument(
        "--devtest_split",
        type=float,
        help="Test size for train-test split. Note that, the test split will be half of this.",
        default=0.1,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Random state for train-test split",
        default=42,
    )

    parser.add_argument(
        "--restructure_directory",
        action="store_true",
    )

    parser.add_argument(
        "--normalize_text",
        action="store_true",
    )

    parser.add_argument(
        "--sampling_rate", type=int, help="Sampling rate for resampling", default=44100
    )

    return parser


def has_russian_kazakh_words(input_string):
    # Check if the string contains Russian or Kazakh words
    return bool(re.search(r"[а-яА-Я]+", input_string))


def resample_audio(fps_src, fps_dst, srs):
    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda args: sf.write(args[1], *lr.load(args[0], sr=args[2])),
                    zip(fps_src, fps_dst, srs),
                ),
                total=len(fps_src),
            )
        )


def restructure_directory(source, destination):
    os.makedirs(os.path.join(destination, "tmp"), exist_ok=True)

    with ThreadPoolExecutor() as executor:
        executor.map(
            lambda args: copyfile(*args),
            [
                (
                    os.path.join(source, fname),
                    os.path.join(destination, "tmp", fname),
                )
                for fname in os.listdir(source)
                if fname.endswith("wav")
            ],
        )


def main(args):
    data = []

    normalizer = IndicNormalizerFactory().get_normalizer("hi")

    for gender in ["male", "female"]:
        path = os.path.join(args.source, gender)
        fnames = os.listdir(path)

        if args.restructure_directory:
            restructure_directory(path, args.destination)

        os.makedirs(os.path.join(args.destination, "wavs"), exist_ok=True)

        if args.sampling_rate != 44100:
            srs = [args.sampling_rate] * len(fnames)
            fps_src = [
                os.path.join(args.destination, "tmp", fn)
                for fn in os.listdir(os.path.join(args.destination, "tmp"))
            ]
            fps_dst = [
                os.path.join(args.destination, "wavs", fn)
                for fn in os.listdir(os.path.join(args.destination, "tmp"))
            ]
            resample_audio(fps_src, fps_dst, srs)
        else:
            with ThreadPoolExecutor() as executor:
                executor.map(
                    lambda args: copyfile(*args),
                    [
                        (
                            os.path.join(args.destination, "tmp", fname),
                            os.path.join(args.destination, "wavs", fname),
                        )
                        for fname in os.listdir(os.path.join(args.destination, "tmp"))
                    ],
                )

        rmtree(os.path.join(args.destination, "tmp"))

        for fname in tqdm(fnames, total=len(fnames)):
            if fname.endswith("txt"):
                stem = fname.split(".")[0]

                with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if has_russian_kazakh_words(content):
                    continue

                content = (
                    content.replace("(", "")
                    .replace(")", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(":", ",")
                    .replace(";", ",")
                    .replace('"', "")
                    .replace("\n", " ")
                    .replace("\t", " ")
                )

                content = re.sub(r"\s+", " ", content)
                content = re.sub(r"([.,?!।])+", r"\1", content)
                content = content.strip(",")
                content = content.strip(".")

                if args.normalize_text:
                    content = normalizer.normalize(content)
                    content = " ".join(indic_tokenize.trivial_tokenize(content, "hi"))

                data.append([stem, content, gender])

    metadata = pd.DataFrame(data, columns=["id", "text", "speaker"])

    metadata.to_csv(
        os.path.join(args.destination, "metadata.csv"),
        sep="$",
        index=False,
        header=False,
    )

    df_train, df_devtest = train_test_split(
        metadata,
        test_size=args.devtest_split,
        random_state=args.random_state,
        stratify=metadata["speaker"],
    )

    df_dev, df_test = train_test_split(
        df_devtest,
        test_size=0.5,
        random_state=args.random_state,
        stratify=df_devtest["speaker"],
    )

    df_train.to_csv(
        os.path.join(args.destination, "metadata_train.csv"),
        sep="$",
        index=False,
        header=False,
    )
    df_dev.to_csv(
        os.path.join(args.destination, "metadata_val.csv"),
        sep="$",
        index=False,
        header=False,
    )
    df_test.to_csv(
        os.path.join(args.destination, "metadata_test.csv"),
        sep="$",
        index=False,
        header=False,
    )


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    main(args)
