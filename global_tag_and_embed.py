# -*- coding: utf-8 -*-
# @Author   : jeffcheng
# @Time     : 2021/9/1 - 15:13
# @Reference: a inference script for single audio, heavily base on demo.py and traintest.py
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import csv

torchaudio.set_audio_backend("soundfile")       # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel
import torchaudio.functional as F

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'


def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    # kaiser_fast
    resampled_waveform = F.resample(
        waveform,
        sr,
        16000,
        lowpass_filter_width=16,
        rolloff=0.85,
        resampling_method="kaiser_window",
        beta=8.555504641634386,
    )
    fbank = torchaudio.compliance.kaldi.fbank(
        resampled_waveform,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )

    n_frames = fbank.shape[0]
    for i in range(0, n_frames, target_length):
        c_fbank = fbank[i : i + target_length, :]
        c_fbank = (c_fbank - (-4.2677393)) / (4.5689974 * 2)
        if c_fbank.shape[0] == target_length:
            yield c_fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audio_path ./0OxlgIitVig.wav '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')

    parser.add_argument("--model_path", type=str, required=True,
                        help="the trained model you want to test")
    parser.add_argument('--audio_path',
                        help='the audio you want to predict, sample rate 16k.',
                        type=str, required=True)

    args = parser.parse_args()

    label_csv = './data/class_labels_indices.csv'       # label and indices for audioset data

    # assume each input spectrogram has 100 time frames
    # 2. load the best model and the weights
    checkpoint_path = args.model_path
    ast_mdl = ASTModel(
        label_dim=527,
        input_tdim=1024,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    print(f"[*INFO] load checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)

    # 1. make feature for predict
    audio_root = args.audio_path
    files = Path(audio_root).glob("*.wav")
    chunk_len = 163840  # based on 1024 mel frames
    rate = 16000

    for file in files:
        print("--------------------------", file)
        sample_pos = 0
        audio_path = Path(file)
        with open(
            Path(Path(audio_path).stem).with_suffix(".csv"), "w", newline=""
        ) as csvfile:
            for feats in make_features(audio_path, mel_bins=128):
                audio_model = audio_model.to(torch.device("cpu"))

                # 3. feed the data feature to model
                feats_data = feats.expand(1, 1024, 128)  # reshape the feature

                audio_model.eval()  # set the eval model
                with torch.no_grad():
                    output = audio_model.forward(feats_data)
                    output = torch.sigmoid(output)
                result_output = output.data.numpy()[0]

                # 4. map the post-prob to label
                labels = load_label(label_csv)

                sorted_indexes = np.argsort(result_output)[::-1]

                # Print audio tagging top probabilities
                print("[*INFO] predice results:")
                top1 = np.array(labels)[sorted_indexes[0]]
                top2 = np.array(labels)[sorted_indexes[1]]
                top3 = np.array(labels)[sorted_indexes[2]]
                spamwriter = csv.writer(
                    csvfile, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                spamwriter.writerow(
                    [
                        sample_pos,
                        sample_pos / rate,
                        top1,
                        top2,
                        top3,
                        result_output[sorted_indexes[0]],
                        result_output[sorted_indexes[1]],
                        result_output[sorted_indexes[2]],
                    ]
                )
                sample_pos += chunk_len
                for k in range(10):
                    print(
                        "{}: {:.4f}".format(
                            np.array(labels)[sorted_indexes[k]],
                            result_output[sorted_indexes[k]],
                        )
                    )
