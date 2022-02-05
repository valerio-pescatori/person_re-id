import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import utils
from metrics import metrics, topKAccuracy
from tcn import TemporalConvNet

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
        self.linear = nn.Linear(
            hidden_size * N_OF_FRAMES + global_features_size, num_classes)

    def forward(self, input):
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
        self.linear = nn.Linear(
            hidden_size * N_OF_FRAMES + global_features_size, num_classes)

    def forward(self, input):
        output = 0
        local_f, global_f = input
        h_t, _ = self.gru(local_f)
        output = torch.zeros((local_f.size(0), num_classes))
        for i, anim in enumerate(h_t):
            linear_input = torch.cat((anim.reshape(-1), global_f[i]))
            output[i] = self.linear(linear_input)
        return output


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels)
        self.linear = nn.Linear(num_channels[-1] + 1, output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        local_f, global_f = x
        local_f = local_f.reshape(
            (-1, input_size, N_OF_FRAMES))
        y1 = self.tcn(local_f)
        linear_in = torch.cat((y1[:, :, -1], global_f), 1)
        return self.linear(linear_in)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(
            hidden_size * N_OF_FRAMES + global_features_size, num_classes)

    def forward(self, input):
        output = 0
        local_f, global_f = input
        h_t, _ = self.rnn(local_f)
        output = torch.zeros((local_f.size(0), num_classes))
        for i, anim in enumerate(h_t):
            linear_input = torch.cat((anim.reshape(-1), global_f[i]))
            output[i] = self.linear(linear_input)
        return output


class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.l1 = nn.Linear(input_size * N_OF_FRAMES +
                            global_features_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        out = self.l1(input)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        return self.l6(out)


class DeepMLP2(nn.Module):
    def __init__(self):
        super(DeepMLP2, self).__init__()
        self.l1 = nn.Linear(input_size * N_OF_FRAMES +
                            global_features_size, 1024)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, num_classes)

    def forward(self, input):
        out = self.l1(input)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        return self.l4(out)


def train(model, optim, criterion, data, target, epochs=10, show_plot=True, save_state=False, load_state=False):
    model_name = model.__class__.__name__
    if(load_state):
        model.load_state_dict(torch.load(
            "model_states/" + model_name + "_state_dict.pt"))

    loss_values = []
    accuracy_values = []
    for e in range(epochs):
        print("Epoch: ", e)
        optim.zero_grad()
        output = model(data)
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
    if(save_state):
        torch.save(model.state_dict(), "model_states/" +
                   model_name + "_state_dict.pt")
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


def test(model, data, target, save_results=True):
    softmax = nn.Softmax(dim=1)
    model_name = model.__class__.__name__
    with torch.no_grad():
        guess = model(data)
        normalized_guess = softmax(guess)
        # rank-1 accuracy, precision, recall and f-1 metrics
        results = metrics(normalized_guess, target)
        # rank-5 accuracy
        rank5 = topKAccuracy(normalized_guess, target, rank=5)
        # rank-10 accuracy
        rank10 = topKAccuracy(normalized_guess, target, rank=10)
        results += (
            "\nrank-5 accuracy \t\t\t" +
            str(rank5) + "\nrank-10 accuracy \t\t\t" + str(rank10)
        )
        if save_results:
            torch.save(guess, "data/" + model_name + "_guess.pt")
            with open("data/" + model_name + "_metrics.txt", "w") as f:
                f.write(results)
                f.close()


if __name__ == "__main__":
    # carico il dataset dai JSON
    local_features, global_features, target = [], [], []
    utils.loadJson(
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
    gru = GRU()
    gru_optim = torch.optim.Adam(gru.parameters(), lr=0.001)
    mlp = DeepMLP()
    mlp_optim = torch.optim.Adam(mlp.parameters(), lr=0.001)
    tcn = TCN(input_size, num_classes, [
              hidden_size, hidden_size, hidden_size, hidden_size])
    tcn_optim = torch.optim.Adam(tcn.parameters(), lr=0.001)
    rnn = RNN()
    rnn_optim = torch.optim.Adam(rnn.parameters(), lr=0.001)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # ##### LSTM #####
    # train(lstm, lstm_optim, criterion,
    #       (train_local_features, train_global_features), train_target)
    # test(lstm, (test_local_features, test_global_features), test_target)

    # ##### GRU #####
    # train(gru, gru_optim, criterion, (train_local_features,
    #       train_global_features), train_target)
    # test(gru, (test_local_features, test_global_features), test_target)

    # ##### TCN #####
    # train(tcn, tcn_optim, criterion, (train_local_features, train_global_features),
    #       train_target, epochs=20, save_state=True, load_state=True)
    # test(tcn, (test_local_features, test_global_features), test_target)

    ##### RNN #####
    # train(rnn, rnn_optim, criterion, (train_local_features, train_global_features),
    #       train_target, epochs=20, save_state=True)
    # test(rnn, (test_local_features, test_global_features), test_target)

    ##### MLP #####
    # train_local_features = train_local_features.reshape((56*3, -1))
    # test_local_features = test_local_features.reshape((56*4, -1))
    # train_local_features = torch.cat(
    #     (train_local_features, train_global_features), 1)
    # test_local_features = torch.cat(
    #     (test_local_features, test_global_features), 1)
    # train(mlp, mlp_optim, criterion, train_local_features,
    #       train_target, epochs=50)
    # test(mlp, test_local_features, test_target)

    ##### MLP2 #####
    # mlp2 = DeepMLP2()
    # mlp2_optim = torch.optim.Adam(mlp2.parameters(), lr=0.001)
    # train(mlp2, mlp2_optim, criterion, train_local_features,
    #       train_target, epochs=80, save_state=True)
    # test(mlp2, test_local_features, test_target)
