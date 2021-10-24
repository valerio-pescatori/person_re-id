import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from pprint import pprint
import numpy as np

# architettura nn:
# 1  lstm layer x local mf
# concateno l'output del lstm layer con le global mf
# 1 dense layer che mi restituisce l'indice dell'animazione


# input del primo layer sono le features locali quindi
# body openness upper
# body openness lower
# bct upper
# bct lower
# bct full
# out-toeing l
# out-toeing r
# hunchback

# re-identification metrics performance metrics
# re-identification ranking

# Hyper-parameters
N_OF_FRAMES = 750
input_size = 188
hidden_size = 512  # arbitrario
num_classes = 56  # numero totale di animazioni


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # definisco la struttura
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size * N_OF_FRAMES, 56)

    def forward(self, input):
        output = 0
        h_t, _ = self.lstm(input)
        # output = self.dense(h_t)
        output = torch.zeros((56, 56))
        for i, anim in enumerate(h_t):
            output[i] = self.dense(anim.reshape(-1))
        return output


def preprocessData(file_path, input_lf, input_gf, target):
    data = None
    with open(file_path, "r") as file:
        data = json.load(file)
    for animation in data["Items"]:  # num_classes animazioni
        anim_local_features = []  # liste di local feature per questa animazione (2D)
        for frame in animation["frames"]:
            frame_local_features = []  # lista di local features per questo frame (1D)
            # appiattisco la lista di local features
            for local_feature in frame.values():
                if type(local_feature) is float:  # se è float lo appendo
                    frame_local_features.append(local_feature)
                else:  # se è lista joino le liste
                    frame_local_features += local_feature
            anim_local_features.append(frame_local_features)
        input_lf.append(anim_local_features)
        input_gf.append([animation["mediaLungPass"]])
        target.append([animation["index"]])


if __name__ == "__main__":
    # preparo training input e target
    train_local_features, train_global_features, train_target = [], [], []
    preprocessData(
        str(Path.cwd().parent) + "\\training.json",
        train_local_features,
        train_global_features,
        train_target,
    )
    train_local_features = torch.tensor(train_local_features)
    # size [num_classes, 750, 188]
    train_global_features = torch.tensor(train_global_features)
    # size [num_classes, 1]
    train_target = torch.tensor(train_target).reshape(-1)
    # size [num_classes, 1]

    ## preparo testing input e target
    test_local_features, test_global_features, test_target = [], [], []
    preprocessData(
        str(Path.cwd().parent) + "\\testing.json",
        test_local_features,
        test_global_features,
        test_target,
    )
    test_local_features = torch.tensor(test_local_features)
    test_global_features = torch.tensor(test_global_features)
    test_target = torch.tensor(test_target).reshape(-1)

    ## istanzio il modello
    lstm = LSTM()
    optim = torch.optim.Adam(lstm.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    ################################# TRAINING #################################
    n_steps = 10
    for step in range(n_steps):
        print("Step: ", step)
        optim.zero_grad()
        output = lstm(train_local_features)
        loss = criterion(output, train_target)
        print("loss: ", loss.item())
        loss.backward()
        optim.step()

    # training completo, ora testo
    with torch.no_grad():
        guess = lstm(test_local_features)
        loss = criterion(guess, test_target)

        # printo esempi a caso
        print("GUESS 1 : \n")
        pprint(guess[12])
        pprint(test_target[12])

        print("\nGUESS 2 : \n")
        pprint(guess[33])
        pprint(test_target[33])

        print("\nGUESS 3 : \n")
        pprint(guess[21])
        pprint(test_target[21])

        print("\nGUESS 4 : \n")
        pprint(guess[6])
        pprint(test_target[6])
        # devo fare 5-ranking qui (?)
