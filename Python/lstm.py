from typing import Iterator
import torch
import torch.nn as nn
import json
from pathlib import Path
from torch.optim import optimizer

from torch.serialization import normalize_storage_type

# architettura nn:
# 1  lstm layer x local mf
# concateno l'output del lstm layer con le global mf
# 1 dense layer che mi restituisce l'indice dell'animazione

# preparo i dati

# input del primo layer sono le features locali quindi
# body openness upper
# body openness lower
# bct upper
# bct lower
# bct full
# out-toeing l
# out-toeing r
# hunchback
# TOT: 8

# input del 1° dense layer è composto da:
# output del layer precedente + vettore delle features globali (solo mediaLungPass)


# Hyper-parameters

# N.B: in fase di developing
# 8 local features
# 250 frames (saranno 3000 poi)
input_size = 8 * 250
hidden_size = 64  # arbitrario
num_classes = 56  # num di animazioni
num_epochs = 3
sequence_length = 8
batch_size = 750  # 3000
learning_rate = 0.001

# forse batch_size = 3000(750)
# e sequenze_length = 8
# https://discuss.pytorch.org/t/batch-size-for-lstm/47619


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # definisco la struttura
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, input):
        h_t = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
        c_t = torch.zeros(input.size(0), hidden_size, dtype=torch.float)
        output = 0
        for frame in input.split(1, dim=1):
            h_t, c_t = self.lstm(frame, (h_t, c_t))
            output = self.linear(h_t)

        return torch.cat(output, dim=1)


def preprocessData(data, input_lmf, input_gmf, target):
    for animation in data:
        anim_lmf = []  # local feature per questa animazione
        for frame in animation["frames"]:
            anim_lmf.append(list(frame.values()))
        input_lmf.append(anim_lmf)
        input_gmf.append([animation["mediaLungPass"]])
        target.append([animation["index"]])


if __name__ == "__main__":
    # preparo i dati
    data = None
    with open(str(Path.cwd().parent) + "\\training.json", "r") as file:
        data = json.load(file)
    # separo local features da global features
    trainLocalFeatures, trainGlobalFeatures, trainTarget = [], [], []
    preprocessData(data["Items"], trainLocalFeatures, trainGlobalFeatures, trainTarget)
    trainLocalFeatures = torch.tensor(trainLocalFeatures)  # size [56, 750, 4]
    trainGlobalFeatures = torch.tensor(trainGlobalFeatures)  # size [56, 1]
    trainTarget = torch.tensor(trainTarget)

    ## preparo il train input e target
    data = None
    with open(str(Path.cwd().parent) + "\\testing.json", "r") as file:
        data = json.load(file)

    testLocalFeatures, testGlobalFeatures, testTarget = [], [], []
    preprocessData(data["Items"], testLocalFeatures, testGlobalFeatures, testTarget)
    testLocalFeatures = torch.tensor(testLocalFeatures)
    testGlobalFeatures = torch.tensor(testGlobalFeatures)
    testTarget = torch.tensor(testTarget)

    ## istanzio il modello
    lstm = LSTM()
    optim = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    ## mi manca il test_input e test_target

    ## training
    n_steps = 10
    for step in range(n_steps):
        print("Step: ", step)

        def closure():
            optim.zero_grad()
            output = lstm(trainLocalFeatures)
            loss = criterion(output, trainTarget)
            print("loss: ", loss.item())
            loss.backward()
            return loss

        optim.step(closure)

        # training completo, ora testo
        # with torch.no_grad()
