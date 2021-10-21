import torch
import torch.nn as nn
import json
from pathlib import Path

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
# 8 local features
# 250 frames (saranno 3000 poi)
sequence_length = 250
input_size = 8
hidden_size = 64  # arbitrario
batch_size = 56  # numero totale di animazioni
# num_epochs = 3

# target deve avere la shape [56, 250, 1] --> ho un risultato per ogni blocco (da 8) di features
# in realtà io ho bisogno di un risultato per ogni blocco da 250*8 di features
# quindi le features devono essere 250*8, il vettore deve essere [1, 56, 250*8] (?)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # definisco la struttura
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, input):  # input deve essere un tensor[1, 250, 8]
        output = 0
        h_t, _ = self.lstm(input)
        output = self.dense(h_t)
        print(output.size())
        return output


def preprocessData(file_path, input_lmf, input_gmf, target):
    data = None
    with open(file_path, "r") as file:
        data = json.load(file)
    for animation in data["Items"]:
        anim_lmf = []  # local feature per questa animazione
        for frame in animation["frames"]:
            anim_lmf.append(list(frame.values()))
        input_lmf.append(anim_lmf)
        input_gmf.append([animation["mediaLungPass"]])
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
    train_local_features = torch.tensor(train_local_features)  # size [56, 250, 8]
    train_global_features = torch.tensor(train_global_features)  # size [56, 1]
    train_target = torch.tensor(train_target)

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
    test_target = torch.tensor(test_target)

    ## istanzio il modello
    lstm = LSTM()
    optim = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    ################################# TRAINING #################################

    # l'input deve avere la seguente shape: [batch_size, sequence_length, input_size]
    # quindi [56, 250, 8]
    n_steps = 10
    for step in range(n_steps):
        print("Step: ", step)
        optim.zero_grad()
        output = lstm(train_local_features)
        loss = criterion(
            output, train_target
        )  # target deve avere la stessa shape dell'input
        print("loss: ", loss.item())
        loss.backward()
        optim.step()

        # training completo, ora testo
        with torch.no_grad():
            guess = lstm(test_local_features)
            loss = criterion()
