import os
import pandas as pd


def formatter(root_path, meta_file, **kwargs):
    file_path = os.path.join(root_path, meta_file)

    df = pd.read_csv(
        file_path,
        sep="$",
        header=None,
        names=["file_id", "text", "speaker_name"],
        encoding="utf-8",
    )

    df["audio_file"] = df["file_id"].apply(
        lambda x: os.path.join(root_path, "wavs", f"{x}.wav")
    )

    df = df.drop(columns=["file_id"])
    df["root_path"] = root_path
    items = df.to_dict("records")

    return items
