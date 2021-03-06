import time
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import utils
from tcn import TemporalConvNet

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
        self.linear = nn.Linear(
            num_channels[-1] + global_features_size, output_size)
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


def train(model, optim, criterion, data, target, epochs=10, save_state=False, load_state=False, ablate=0):
    start_time = time.time()
    model_name = model.__class__.__name__
    last_epoch = 0
    if(load_state):
        checkpoint = torch.load(
            "model_states/"+model_name+"_ablate"+str(ablate)+"_state.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    loss_values = []
    accuracy_values = []
    for e in range(last_epoch, last_epoch + epochs):
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
    if save_state:
        torch.save({
            'epoch': last_epoch + epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss_values[-1],
        }, "model_states/" + model_name + "_ablate" + str(ablate) + "_state.pt")
    plt.clf()
    x = [_ for _ in range(last_epoch, last_epoch + epochs)]
    plt.plot(x, loss_values)
    plt.title(model_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("data/plots/" + model_name +
                "_ablate" + str(ablate) + "_loss.png")

    plt.clf()
    plt.plot(x, accuracy_values)
    plt.title(model_name)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("data/plots/" + model_name + "_ablate" +
                str(ablate) + "_accuracy.png")
    print("\n--- Training completed in %s seconds ---" %
          (time.time() - start_time))


def test(model, data, target, ablate=0):
    start_time = time.time()
    softmax = nn.Softmax(dim=1)
    model_name = model.__class__.__name__
    with torch.no_grad():
        guess = model(data)
        normalized_guess = softmax(guess)
        # rank-1 accuracy, precision, recall and f-1 metrics
        results = utils.metrics(normalized_guess, target)
        # rank-5 accuracy
        rank5 = utils.topKAccuracy(normalized_guess, target, rank=5)
        # rank-10 accuracy
        rank10 = utils.topKAccuracy(normalized_guess, target, rank=10)
        results += (
            "\nrank-5 accuracy \t\t\t" +
            str(rank5) + "\nrank-10 accuracy \t\t\t" + str(rank10)
        )
        # confusion matrix
        utils.confusionMatrix(
            target, guess, model_name + "_ablate"+str(ablate))
        # save metrics
        with open("data/" + model_name + "_ablate" + str(ablate) + "_metrics.txt", "w") as f:
            f.write(results)
            f.close()
    print("\n--- Testing completed in %s seconds ---" %
          (time.time() - start_time))


def runTests(abl, save_states, load_states):
    abl_from, abl_to = 0, 1
    if (abl):
        abl_to = 4
    for ablate in range(abl_from, abl_to):
        # carico il dataset dai JSON
        local_features, global_features, target = utils.loadJson(
            str(Path.cwd().parent) + "\\Data\\", ablate=ablate)
        input_size = local_features.size(2)

        # splitto il dataset in train e test
        train_local_features, test_local_features = torch.split(
            local_features, [num_classes * 3, num_classes * 4])
        train_global_features, test_global_features = torch.split(
            global_features, [num_classes * 3, num_classes * 4])
        train_target, test_target = torch.split(
            target, [num_classes * 3, num_classes * 4])

        # istanzio i modelli
        lstm = LSTM()
        lstm_optim = torch.optim.Adam(lstm.parameters(), lr=0.001)
        gru = GRU()
        gru_optim = torch.optim.Adam(gru.parameters(), lr=0.001)
        mlp = DeepMLP()
        mlp_optim = torch.optim.Adam(mlp.parameters(), lr=0.001)
        mlp2 = DeepMLP2()
        mlp2_optim = torch.optim.Adam(mlp2.parameters(), lr=0.001)
        tcn = TCN(input_size, num_classes, [
            hidden_size, hidden_size, hidden_size, hidden_size])
        tcn_optim = torch.optim.Adam(tcn.parameters(), lr=0.001)
        rnn = RNN()
        rnn_optim = torch.optim.Adam(rnn.parameters(), lr=0.001)

        # loss function
        criterion = nn.CrossEntropyLoss()

        #### LSTM #####
        train(lstm, lstm_optim, criterion,
              (train_local_features, train_global_features), train_target, ablate=ablate, save_state=save_states, load_state=load_states)
        test(lstm, (test_local_features, test_global_features),
             test_target, ablate=ablate)

        ##### GRU #####
        train(gru, gru_optim, criterion, (train_local_features,
                                          train_global_features), train_target, ablate=ablate, save_state=save_states, load_state=load_states)
        test(gru, (test_local_features, test_global_features),
             test_target, ablate=ablate)

        ##### TCN #####
        train(tcn, tcn_optim, criterion, (train_local_features, train_global_features),
              train_target, epochs=60, ablate=ablate, save_state=save_states, load_state=load_states)
        test(tcn, (test_local_features, test_global_features),
             test_target, ablate=ablate)

        ##### RNN #####
        train(rnn, rnn_optim, criterion, (train_local_features, train_global_features),
              train_target, epochs=20, ablate=ablate, save_state=save_states, load_state=load_states)
        test(rnn, (test_local_features, test_global_features),
             test_target, ablate=ablate)

        #### MLP #####
        # flattening della sequenza per i MLP
        train_local_features = train_local_features.reshape(
            (train_local_features.size(0), -1))
        test_local_features = test_local_features.reshape(
            (test_local_features.size(0), -1))
        train_local_features = torch.cat(
            (train_local_features, train_global_features), 1)
        test_local_features = torch.cat(
            (test_local_features, test_global_features), 1)

        train(mlp, mlp_optim, criterion, train_local_features,
              train_target, epochs=100, ablate=ablate, save_state=save_states, load_state=load_states)
        test(mlp, test_local_features, test_target, ablate=ablate)

        #### MLP2 #####
        train(mlp2, mlp2_optim, criterion, train_local_features,
              train_target, epochs=80, ablate=ablate, save_state=save_states, load_state=load_states)
        test(mlp2, test_local_features, test_target, ablate=ablate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", "-a",
                        action="store_true",
                        help="Add --ablation/-a if you want to to execute ablation test",
                        default=False)
    parser.add_argument("--save_states", "-ss",
                        action="store_true",
                        help="Add --save_states/-ss if you want to save all nn models states to /model_states/",
                        default=False)
    parser.add_argument("-load_states", "-ls",
                        action="store_true",
                        help="Add --load_states/-ls if you want to load all nn models states from /model_states/",
                        default=False)
    args = parser.parse_args()

    start_time = time.time()
    runTests(args.ablation, args.save_states, args.load_states)
    print("\n--- Total time elapsed: %s seconds ---" %
          (time.time() - start_time))
