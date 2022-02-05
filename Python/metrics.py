from statistics import mode
from sklearn.metrics import classification_report, top_k_accuracy_score,  multilabel_confusion_matrix
import torch
import seaborn as sn
import matplotlib.pyplot as plt


def topKAccuracy(guess, target, rank=1):
    return round(top_k_accuracy_score(target, guess, k=rank, normalize=True), 2)


def metrics(guess, target):
    pred = torch.zeros((guess.size(0)), dtype=torch.long)
    for i, el in enumerate(guess):
        pred[i] = torch.argmax(el)
    return classification_report(target, pred, zero_division=0)


if __name__ == "__main__":
    models = {"GRU", "LSTM", "RNN", "DeepMLP", "DeepMLP2", "TCN"}
    target = torch.load("data/target.pt")
    for model in models:
        guess = torch.load("data/" + model + "_guess.pt")
        matrix = [[0 for _ in range(56)] for _ in range(56)]
        for i, element in enumerate(target):
            col = torch.argmax(guess[i])
            matrix[element.item()][col] += 1
        plt.clf()
        ax = sn.heatmap(matrix, cmap="YlGnBu")
        plt.title(model + " Confusion Matrix")
        plt.xlabel("Predicted values")
        plt.ylabel("Actual values")
        plt.savefig("data/confusion/" + model + "_cm.png", dpi=600)
