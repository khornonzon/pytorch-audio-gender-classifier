from WavDataLoader import WavDataset
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
import librosa

class GenderClassifier:
    def __init__(self):
        wds_train = WavDataset()
        self.wds_dl = DataLoader(wds_train, batch_size=32, shuffle=True)
        self.model =  torch.nn.Sequential(nn.Conv2d(1, 1, 3),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Conv2d(1, 1, 3),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(7440,128),
                            nn.ReLU(),
                            nn.Linear(128,1),
                            nn.Sigmoid()
                            )
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def accuracy(self, outs, labels):
             result = 0
             for i in range(len(outs)):
                if (outs[i] > 0.5) == labels[i]:
                    result+=1
             return result/len(outs)
    
    def train_one_epoch(self, epoch_index):
     epoch_losses = []
     epoch_accuracies = []
     for i, data in enumerate(self.wds_dl):
        inputs, labels = data
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs.float(), labels.float())
        epoch_accuracies.append(self.accuracy(outputs, labels))
        loss.backward()
        self.optimizer.step()
        epoch_losses.append(loss.item())
     return np.mean(epoch_losses), np.mean(epoch_accuracies)
    
    def train(self, epochs):
        for i in range(epochs):
            print(f'{i} epoch')
            loss, accuracy_epoch = self.train_one_epoch(i)
            print(loss, accuracy_epoch)

    def predict(self, wav_path):
        waveform, sample_rate = librosa.load(wav_path)
        melspectr = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        new_wav = np.pad(melspectr,((0,0),(0, 1000-melspectr.shape[1])), mode='constant',constant_values=((0,0),(0,0)))
        tensor_wav = (torch.tensor(np.array(new_wav))/(np.array(new_wav).max())*255).view(1,128,1000)
        out = self.model(tensor_wav)
        if out >0.5:
            return out, 'this is a woman'
        else:
            return out, 'this is a man'
        

gc = GenderClassifier()
gc.train(50)
print(gc.predict('test\\test\\0a2df9e11b2064f934fc30036888166f.wav'))



        
