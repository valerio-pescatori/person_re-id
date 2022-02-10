import json
from pathlib import Path
from sklearn.metrics import classification_report, top_k_accuracy_score
import torch
import seaborn as sn
import matplotlib.pyplot as plt
import time


def loadJson(path, ablate=0):
    start_time = time.time()
    # leggo i json dal path e ritono i tensor di local, global e target
    loc = torch.empty([392, 750, 188])
    glob = torch.empty([392, 1])
    targ = torch.empty(392)
    for i in range(3):
        with open(path + "data" + str(i) + ".json") as file:
            data = json.load(file)
        for anim_i, anim in enumerate(data["Items"]):  # 392
            # local
            for frame_i, frame in enumerate(anim["frames"]):  # 750
                pos = frame.pop("positions")
                acc = frame.pop("accelerations")
                vel = frame.pop("velocities")
                if ablate == 1:
                    frame.pop("hunchback")
                    frame.pop("outToeingL")
                    frame.pop("outToeingR")
                if ablate == 2:
                    pos = []
                    acc = []
                if ablate == 3:
                    frame.pop("bodyOpennessU")
                    frame.pop("bodyOpennessL")
                    frame.pop("bctU")
                    frame.pop("bctL")
                    frame.pop("bctF")
                temp = pos + acc + vel
                temp += frame.values()
                if (loc.shape != torch.Size([392, 750, len(temp)])):
                    loc, _ = torch.split(
                        loc, [len(temp), 188-len(temp)], dim=2)
                loc[anim_i][frame_i] = torch.tensor(temp)
            # global
            loc[anim_i] = torch.tensor(anim["mediaLungPass"])
            # target
            targ[anim_i] = anim["index"]
    print("\n---NEW: %s seconds ---" % (time.time() - start_time))
    print(loc.shape, glob.shape, targ.shape)
    return loc, glob, targ


def topKAccuracy(guess, target, rank=1):
    return round(top_k_accuracy_score(target, guess, k=rank, normalize=True), 2)


def metrics(guess, target):
    pred = torch.zeros((guess.size(0)), dtype=torch.long)
    for i, el in enumerate(guess):
        pred[i] = torch.argmax(el)
    return classification_report(target, pred, zero_division=0)


def confusionMatrix(target, guess, model_name):
    matrix = [[0 for _ in range(56)] for _ in range(56)]
    for i, element in enumerate(target):
        col = torch.argmax(guess[i])
        matrix[element.item()][col] += 1
    plt.clf()
    sn.heatmap(matrix, cmap="YlGnBu")
    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.savefig("data/cm/" + model_name + "_cm.png", dpi=600)


if __name__ == "__main__":
    # preparo json per lo studio ablativo

    # carico json

    loadJson(str(Path.cwd().parent) + "\\Data\\", ablate=1)
    loadJson(str(Path.cwd().parent) + "\\Data\\", ablate=2)
    loadJson(str(Path.cwd().parent) + "\\Data\\", ablate=3)
