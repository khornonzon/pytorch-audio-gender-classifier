from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
import pandas as pd
import librosa
import tqdm
import numpy as np
from torchsummary import summary


class WavDataset(Dataset):
    def __init__(self, data_folder= 'train/train'):
        self.data = pd.read_csv('targets.tsv', sep='\t', header = None)
        self.file_names, self.targets = self.data[0][:500], self.data[1][:500]
        print(self.file_names)
        self.wavs = []
        self.new_wavs = []
        for i, name in enumerate(self.file_names):
            print(i)
            waveform, sample_rate = librosa.load('train/train/'+name+'.wav')
            self.wavs.append(librosa.feature.melspectrogram(y=waveform, sr=sample_rate))
        for wav in self.wavs:
            new_wav = np.pad(wav,((0,0),(0, 1000-wav.shape[1])), mode='constant',constant_values=((0,0),(0,0)))
            self.new_wavs.append(new_wav)
        self.wavs = self.new_wavs
        self.tensor_wavs = torch.tensor(np.array(self.wavs))/(np.array(self.wavs).max())*255
        self.tensor_targets = np.array(self.targets)
        self.tensor_targets = torch.tensor(self.tensor_targets) 
    def __getitem__(self, ix):
        return self.tensor_wavs[ix].view(1,128,1000), self.tensor_targets[ix].view(1)
    def __len__(self):
        return self.tensor_targets.shape[0]



