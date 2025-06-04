from torch.utils.data import DataLoader, Dataset
from tokenizer.tokenizer import Tokenizer
import torch
import random
import os
from utils.utils import collate_with_PAD
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio as ta
import torch

import torch
import librosa
import numpy as np

def extract_audio_features(wav_file: str) -> torch.Tensor:
    if not isinstance(wav_file, str):
        raise TypeError(f"Expected string for wav_file")
    
    # 读取音频
    waveform, sr = librosa.load(wav_file, sr=16000)  # 统一采样率
    # 提取梅尔谱特征
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sr, 
        n_fft=400, 
        hop_length=160, 
        n_mels=80
    )
    # 对数梅尔谱
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    features = torch.from_numpy(log_mel.T).float()  # shape: (L, 80)
    features = (features - features.mean()) / (features.std() + 1e-6)
    if not isinstance(features, torch.Tensor):
        raise TypeError("Return value must be torch.Tensor")
    if features.ndim != 2 or features.shape[1] != 80:
        raise ValueError("Returned tensor shape must be (L, 80)")
    return features


class BZNSYP(Dataset):
    def __init__(self, wav_file, text_file, tokenizer):
        self.tokenizer = tokenizer
        self.wav2path = {}
        self.wav2text = {}
        self.ids = []

        with open(wav_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    id = parts[0]
                    self.ids.append(id)
                    path = "./dataset/" + parts[1]
                    self.wav2path[id] = path
                else:
                    raise ValueError(f"Invalid line format: {line}")

        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    id = parts[0]
                    pinyin_list = parts[1].split(" ")
                    self.wav2text[id] = self.tokenizer(["<sos>"]+pinyin_list+["<eos>"])
                else:
                    raise ValueError(f"Invalid line format: {line}")
    
    def __len__(self):
        return len(self.wav2path)
    
    def __getitem__(self, index):
        id = list(self.wav2path.keys())[index]
        path = self.wav2path[id]
        text = self.wav2text[id]
        return id, extract_audio_features(path), text
    

def get_dataloader(wav_file, text_file, batch_size, tokenizer, shuffle=True):
    dataset = BZNSYP(wav_file, text_file, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_with_PAD
    )
    return dataloader