import torch
import torch.nn as nn
import json
from pathlib import Path
import matplotlib.pyplot as plt

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
        output = torch.zeros((56, 56))
        for i, anim in enumerate(h_t):
            dense_input = torch.cat((anim.reshape(-1), global_f[i]))
            output[i] = self.dense(dense_input)
        return output


def preprocessData(file_path, input_lf, input_gf, target):
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
    optim = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    ################################# TRAINING #################################
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
    # plt.savefig("Python/loss.png")

    plt.plot(x, accuracy)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()
    # plt.savefig("Python/accuracy.png")

    # training completo, ora testo
    with torch.no_grad():
        guess = lstm((test_local_features, test_global_features))
        loss = criterion(guess, test_target)

        torch.save(guess, "guess.pt")
        torch.save(test_target, "target.pt")
        # misuro l'accuracy
        corrette = 0
        for i in range(num_classes):
            if torch.argmax(guess[i]) == test_target[i]:
                corrette += 1

        print("loss: ", loss.item())
        print("Risposte corrette: ", corrette, "/", num_classes)
        print(round(corrette / num_classes * 100, 2), "%")

        print("\n\nEsempio risultato:")
        print(guess[13])
        print("target: ", test_target[13])
        print("argmax: ", torch.argmax(guess[13]))
