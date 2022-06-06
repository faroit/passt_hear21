import torch
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import csv

from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings

torchaudio.set_audio_backend("soundfile")  # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import torchaudio.functional as F


def load_label(label_csv):
    with open(label_csv, "r") as f:
        reader = csv.reader(f, delimiter=",")
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Example of parser:"
        "python inference --audio_path ./0OxlgIitVig.wav "
        "--model_path ./pretrained_models/audioset_10_10_0.4593.pth"
    )

    parser.add_argument(
        "audio_path",
        help="the audio you want to predict, sample rate 16k.",
        type=str,
    )

    args = parser.parse_args()

    model = load_model(mode="logits")
    embed = load_model(mode="embed_only")

    # audio = torch.ones((1, 32000 * seconds)) * 0.5

    label_csv = "./class_labels_indices.csv"  # label and indices for audioset data
    labels = load_label(label_csv)

    files = Path(args.audio_path).glob("*.wav")
    model_rate = 32000
    sample_pos = 0

    for file in files:
        with open(
            Path(Path(file).stem).with_suffix(".csv"), "w", newline=""
        ) as csvfile:

            waveform, sr = torchaudio.load(file)
            # kaiser_fast
            resampled_waveform = torchaudio.transforms.Resample(
                sr,
                model_rate,
            )(waveform)

            with torch.no_grad():
                embeddings = get_scene_embeddings(resampled_waveform, embed)
                pred = get_scene_embeddings(resampled_waveform, model)
                output = torch.sigmoid(pred)

            # save embeddings
            # save csvs
            result_output = output.data.numpy()[0]
            sorted_indexes = np.argsort(result_output)[::-1]

            top1 = np.array(labels)[sorted_indexes[0]]
            top2 = np.array(labels)[sorted_indexes[1]]
            top3 = np.array(labels)[sorted_indexes[2]]
            spamwriter = csv.writer(
                csvfile, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            spamwriter.writerow(
                [
                    sample_pos,
                    sample_pos / model_rate,
                    top1,
                    top2,
                    top3,
                    result_output[sorted_indexes[0]],
                    result_output[sorted_indexes[1]],
                    result_output[sorted_indexes[2]],
                ]
            )
            print(embeddings.shape)
            np.save(Path(Path(file).stem).with_suffix(".npy"), embeddings.detach().numpy()) 
            for k in range(3):
                print(
                    "{}: {:.4f}".format(
                        np.array(labels)[sorted_indexes[k]],
                        result_output[sorted_indexes[k]],
                    )
                )