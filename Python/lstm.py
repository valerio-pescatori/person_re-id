import torch
import torch.nn as nn
import json
from pathlib import Path
from pprint import pprint

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
    local_features = []
    global_features = []
    ## struttura finale di local  = [ [val1, ..., valx], ..., [val1, ..., valx] ]
    for animation in data:
        anim_lf = []  # local feature per questa animazione
        for frame in animation["frames"]:
            anim_lf.append(list(frame.values()))
        local_features.append(anim_lf)
        global_features.append([animation["mediaLungPass"]])

    local_features = torch.tensor(local_features)
    # print(local_features.item())
