import torch
from torch.functional import norm
import torch.nn as nn
import json
from pathlib import Path
import matplotlib.pyplot as plt
from metrics import topKAccuracy
from metrics import metrics

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
SAVE_RESULTS = True
input_size = 188
# input_size = 8 + (5 * 9)  # 8 --> ot, hb, bo, bct. 5*3 --> pos, vel e acc per 5 joints
hidden_size = 512
num_classes = 56  # numero totale di animazioni
global_features_size = 1


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # definisco la struttura
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size * N_OF_FRAMES + global_features_size, 56)

    def forward(self, input):
        output = 0
        local_f, global_f = input
        h_t, _ = self.lstm(local_f)
        # output = self.dense(h_t)
        output = torch.zeros((local_f.size(0), num_classes))
        for i, anim in enumerate(h_t):
            dense_input = torch.cat((anim.reshape(-1), global_f[i]))
            output[i] = self.dense(dense_input)
        return output


def preprocessData(file_path):
    localf = []
    globalf = []
    target = []
    data = None
    with open(file_path, "r") as file:
        data = json.load(file)
    for animation in data["Items"]:  # num_classes animazioni
        anim_local_features = []  # liste di local feature per questa animazione
        for frame in animation["frames"]:
            frame_local_features = []  # lista di local features per questo frame
            # appiattisco la lista di local features
            for local_feature in frame.values():
                if type(local_feature) is float:  # se è float lo appendo
                    frame_local_features.append(local_feature)
                else:  # se è lista joino le liste
                    frame_local_features += local_feature
            anim_local_features.append(frame_local_features)
        localf.append(anim_local_features)
        globalf.append([animation["mediaLungPass"]])
        target.append(animation["index"])
    return localf, globalf, target


def loadJson(loc, glob, targ):
    ## divido i samples tra training e testing
    ## 3 per training 4 per testing
    for i in range(3):
        t = preprocessData(str(Path.cwd().parent) + "\\Data\\data" + str(i) + ".json")
        loc += t[0]
        glob += t[1]
        targ += t[2]
    return loc, glob, targ


if __name__ == "__main__":
    local_features, global_features, target = [], [], []
    loadJson(
        local_features,
        global_features,
        target,
    )
    local_features = torch.tensor(local_features)
    global_features = torch.tensor(global_features)
    target = torch.tensor(target)

    train_local_features, test_local_features = torch.split(
        local_features, [56 * 3, 56 * 4]
    )
    train_global_features, test_global_features = torch.split(
        global_features, [56 * 3, 56 * 4]
    )
    train_target, test_target = torch.split(target, [56 * 3, 56 * 4])

    ## istanzio il modello
    lstm = LSTM()
    optim = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    # ################################# TRAINING #################################
    loss_values = []
    accuracy = []
    epochs = 10
    for e in range(epochs):
        print("Epoch: ", e)
        optim.zero_grad()
        output = lstm((train_local_features, train_global_features))
        loss = criterion(output, train_target)
        loss_values.append(loss.item())
        loss.backward()
        optim.step()

        # valori per plot
        corrette = 0
        print("loss: ", loss.item())
        for i in range(num_classes):
            if torch.argmax(output[i]) == train_target[i]:
                corrette += 1
        accuracy.append(round(corrette / num_classes * 100, 2))
    x = [_ for _ in range(epochs)]
    plt.plot(x, loss_values)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    plt.plot(x, accuracy)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()

    # training completo, ora testo
    with torch.no_grad():
        guess = lstm((test_local_features, test_global_features))
        normalizedGuess = softmax(guess)
        if SAVE_RESULTS:
            torch.save(guess, "data/guess.pt")
            torch.save(test_target, "data/target.pt")
        # rank-1 accuracy, precision, recall and f-1 metrics
        results = metrics(normalizedGuess, test_target)
        # rank-5 accuracy
        rank5 = topKAccuracy(normalizedGuess, test_target, rank=5)
        # rank-10 accuracy
        rank10 = topKAccuracy(normalizedGuess, test_target, rank=10)
        results += (
            "\nrank-5 accuracy " + str(rank5) + "\nrank-10 accuracy " + str(rank10)
        )
        with open("data/metrics.txt", "w") as f:
            f.write(results)
            f.close()
        print(results)
