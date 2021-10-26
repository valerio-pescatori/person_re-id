from sklearn.base import _pprint
from sklearn.metrics import classification_report
import torch
from pprint import pprint


def metrics():
    guess = torch.load("guess.pt")
    target = torch.load("target.pt")
    pred = torch.zeros((56), dtype=torch.long)
    for i, el in enumerate(guess):
        pred[i] = torch.argmax(el)

    metrics = classification_report(target, pred, zero_division=0)
    with open("metrics.txt", "w") as f:
        f.write(metrics)
        f.close()


if __name__ == "__main__":
    metrics()
