from matplotlib.pyplot import get_current_fig_manager
from numpy import true_divide
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score
import torch
from torch._C import merge_type_from_type_comment
from torch.functional import norm


def topKAccuracy(guess, target, rank=1):
    return round(top_k_accuracy_score(target, guess, k=rank, normalize=True), 2)


def metrics(guess, target):
    pred = torch.zeros((guess.size(0)), dtype=torch.long)
    for i, el in enumerate(guess):
        pred[i] = torch.argmax(el)
    return classification_report(target, pred, zero_division=0)


if __name__ == "__main__":
    s = torch.nn.Softmax(dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    guess = torch.load("data/guess.pt")
    target = torch.load("data/target.pt")
    guess = s(guess)

    # rank-1 accuracy, precision, recall and f-1 metrics
    results = metrics(guess, target)
    # rank-5 accuracy
    rank5 = topKAccuracy(guess, target, rank=5)
    # rank-10 accuracy
    rank10 = topKAccuracy(guess, target, rank=10)
    results += "\nrank-5 accuracy " + str(rank5) + "\nrank-10 accuracy " + str(rank10)
    with open("data/metrics.txt", "w") as f:
        f.write(results)
        f.close()
    print(results)
