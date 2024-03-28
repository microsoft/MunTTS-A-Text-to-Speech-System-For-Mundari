import os
import argparse
import numpy as np
import pandas as pd
from pymcd.mcd import Calculate_MCD
from joblib import Parallel, delayed


def get_arg_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="calculate mel-cepstral distortion (MCD) between generated and groundtruth audios with SPTK-based mcep."
    )
    parser.add_argument(
        "--gen_wavdir",
        type=str,
        required=True,
        help="directory including generated wav files.",
    )
    parser.add_argument(
        "--gt_wavdir",
        type=str,
        required=True,
        help="directory including groundtruth wav files.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="metadata csv file including speaker id and utterance id.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="all",
        help="speaker id to be evaluated. if 'all', calculate MCD for all speakers.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=22050,
        help="sampling rate of wav files.",
    )
    return parser


def main(args):
    df = pd.read_csv(args.metadata_file, sep="$", names=["file_id", "text", "speaker"])

    if args.speaker != "all":
        df = df[df["speaker"] == args.speaker]

    gt_wav_list = [
        os.path.join(args.gt_wavdir, f"{fname}.wav")
        for fname in df["file_id"].values.tolist()
        if os.path.exists(os.path.join(args.gt_wavdir, f"{fname}.wav"))
    ]
    gen_wav_list = [
        os.path.join(args.gen_wavdir, f"test_{fname}.wav")
        for fname in df["file_id"].values.tolist()
        if os.path.exists(os.path.join(args.gen_wavdir, f"test_{fname}.wav"))
    ]

    assert len(gt_wav_list) == len(
        gen_wav_list
    ), "number of wav files is different\nLength of gt_wav_list: {}, Length of gen_wav_list: {}".format(
        len(gt_wav_list), len(gen_wav_list)
    )

    mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")
    mcd_toolbox.SAMPLING_RATE = args.sampling_rate

    mcds = Parallel(n_jobs=-1)(
        [
            delayed(mcd_toolbox.calculate_mcd)(ref, syn)
            for ref, syn in zip(gt_wav_list, gen_wav_list)
        ]
    )
    print(
        f" | > Mean MCD for {args.speaker} with {len(mcds)} samples: {np.mean(mcds):.2f} \pm {np.std(mcds):.2f}"
    )


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
