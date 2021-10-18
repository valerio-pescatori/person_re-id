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
# 750 frames (saranno 3000 poi)
input_size = 8 * 750
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


if __name__ == "__main__":
    # preparo i dati
    data = None
    with open(str(Path.cwd().parent) + "\\training_data.json", "r") as file:
        data = json.load(file)

    ## process data
    # devo separare local features da global features
    data = data["Items"]
    train_input = []
    train_target = torch.tensor([i for i in range(56)])
    global_features = []
    for animation in data:
        anim_lf = []  # local feature per questa animazione
        for frame in animation["frames"]:
            anim_lf.append(list(frame.values()))
        train_input.append(anim_lf)
        global_features.append([animation["mediaLungPass"]])

    train_input = torch.tensor(train_input)  # size [56, 750, 4]
    global_features = torch.tensor(global_features)  # size [56, 1]

    ## istanzio il modello
    lstm = LSTM()
    optim = torch.optim.Adam(nn.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    ## mi manca il test_input e test_target

    ## training
    n_steps = 10
    for step in range(n_steps):
        print("Step: ", step)

        def closure():
            optim.zero_grad()
            output = lstm(train_input)
            loss = criterion(output, train_target)
            print("loss: ", loss.item())
            loss.backward()
            return loss

        optim.step(closure)

        # training completo, ora testo
        # with torch.no_grad():
