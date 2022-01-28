from threading import local
import matplotlib.pyplot as plt
from sklearn import impute
import torch
import torch.nn as nn

import helper as h
from metrics import metrics, topKAccuracy

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
SAVE_RESULTS = True #se True salva i risultati del testing e i relativi target in due file "guess.pt" e "target.pt"
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
        self.linear = nn.Linear(hidden_size * N_OF_FRAMES + global_features_size, num_classes)

    def forward(self, input, batch_size):
        output = 0
        local_f, global_f = input
        h_t, _ = self.lstm(local_f)
        output = torch.zeros((local_f.size(0), num_classes))
        for i, anim in enumerate(h_t):
            linear_input = torch.cat((anim.reshape(-1), global_f[i]))
            output[i] = self.linear(linear_input)
        return output

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size * N_OF_FRAMES + global_features_size, num_classes)

    def forward(self, input, batch_size):
        output = 0
        local_f, global_f = input
        h_t, _ = self.gru(local_f)
        output = torch.zeros((local_f.size(0), num_classes))
        for i, anim in enumerate(h_t):
            linear_input = torch.cat((anim.reshape(-1), global_f[i]))
            output[i] = self.linear(linear_input)
        return output

class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()  
        self.l1 = nn.Linear(input_size * N_OF_FRAMES + global_features_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, input, batch_size):
        local_f, global_f = input
        local_f = local_f.reshape((batch_size, -1))
        features = torch.cat( (local_f, global_f), 1)

        out = self.l1(features)
        out = self.relu(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        return self.l3(out)
         

def train(model, optim, criterion, local_features, global_features, target, batch_size=56*3, epochs=10, show_plot=True):
    model_name = model.__class__.__name__
    loss_values = []
    accuracy_values = []
    for e in range(epochs):
        print("Epoch: ", e)
        optim.zero_grad()
        output = model((local_features, global_features), batch_size)
        loss = criterion(output, target)
        loss_values.append(loss.item())
        loss.backward()
        optim.step()

        # misuro loss e accuracy
        corrette = 0
        print(model_name, "loss: ", loss.item())
        for i in range(num_classes):
            if torch.argmax(output[i]) == target[i]:
                corrette += 1
        accuracy_values.append(round(corrette / num_classes * 100, 2))
    if show_plot:
        x = [_ for _ in range(epochs)]
        plt.plot(x, loss_values)
        plt.title(model_name)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()

        plt.plot(x, accuracy_values)
        plt.title(model_name)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()

def test(model, local_features, global_features, target, batch_size=56*4, save_results=True):
    softmax = nn.Softmax(dim=1)
    model_name = model.__class__.__name__

    with torch.no_grad():
        guess = model((local_features, global_features), batch_size)
        normalized_guess = softmax(guess)
        if save_results:
            torch.save(guess, "data/" + model_name + "_guess.pt")
            torch.save(target, "data/" + model_name + "_target.pt")
        # rank-1 accuracy, precision, recall and f-1 metrics
        results = metrics(normalized_guess, target)
        # rank-5 accuracy
        rank5 = topKAccuracy(normalized_guess, target, rank=5)
        # rank-10 accuracy
        rank10 = topKAccuracy(normalized_guess, target, rank=10)
        results += (
            "\nrank-5 accuracy \t\t\t" + str(rank5) + "\nrank-10 accuracy \t\t\t" + str(rank10)
        )
        if(save_results):
            with open("data/" + model_name + "_metrics.txt", "w") as f:
                f.write(results)
                f.close()

if __name__ == "__main__":
    # carico i dataset dal JSON
    local_features, global_features, target = [], [], []
    h.loadJson(
        local_features,
        global_features,
        target,
    )
    
    # converto in tensor
    local_features = torch.tensor(local_features)
    global_features = torch.tensor(global_features)
    target = torch.tensor(target)

    # splitto il dataset in train e test
    train_local_features, test_local_features = torch.split(
        local_features, [56 * 3, 56 * 4]
    )
    train_global_features, test_global_features = torch.split(
        global_features, [56 * 3, 56 * 4]
    )
    train_target, test_target = torch.split(target, [56 * 3, 56 * 4])

    # istanzio i modelli
    lstm = LSTM()
    lstm_optim = torch.optim.Adam(lstm.parameters(), lr=0.001)
    gru=GRU()
    gru_optim = torch.optim.Adam(gru.parameters(), lr=0.001)
    mlp = DeepMLP()
    mlp_optim = torch.optim.Adam(mlp.parameters(), lr=0.001)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # train(lstm, lstm_optim, criterion, train_local_features, train_global_features, train_target)
    # test(lstm, test_local_features, test_global_features, test_target)

    train(gru, gru_optim, criterion, train_local_features, train_global_features, train_target)
    test(gru, test_local_features, test_global_features, test_target)

    # train(mlp, mlp_optim, criterion, train_local_features, train_global_features, train_target, epochs=500)
    # test(mlp, test_local_features, test_global_features, test_target)