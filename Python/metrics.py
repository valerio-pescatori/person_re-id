from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
import torch


def topKAccuracy(guess, target, rank=1):
    return round(top_k_accuracy_score(target, guess, k=rank, normalize=True), 2)


def metrics(guess, target):
    pred = torch.zeros((guess.size(0)), dtype=torch.long)
    for i, el in enumerate(guess):
        pred[i] = torch.argmax(el)
    return classification_report(target, pred, zero_division=0)


if __name__ == "__main__":
    models = {"GRU", "LSTM", "RNN", "DeepMLP", "DeepMLP2", "TCN"}
