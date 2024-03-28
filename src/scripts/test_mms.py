import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from transformers import VitsModel, AutoTokenizer
from indicnlp.transliterate import unicode_transliterate

xliterator = unicode_transliterate.UnicodeIndicTransliterator()

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="facebook/mms-tts-unr")
    parser.add_argument("-t", "--text_file", type=str, rqequired=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    return parser


def main(args):
    model = VitsModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_csv(args.text_file, sep="$", names=["file_id", "text", "gender"])
    df = df[df["gender"] == "male"]  ## supports only male speaker for now

    df["transliterated_text"] = df["text"].apply(
        lambda x: xliterator.transliterate(x.strip(), "hi", "or")
    )

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row = row.to_dict()
        inputs = tokenizer(row["transliterated_text"], return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs).waveform[0]

        wavfile.write(
            os.path.join(args.output_dir, f"test_{row['file_id']}.wav"),
            rate=model.config.sampling_rate,
            data=output.cpu().numpy(),
        )


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
