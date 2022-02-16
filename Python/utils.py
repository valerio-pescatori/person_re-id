import json
from sklearn.metrics import classification_report
import torch
import seaborn as sn
import matplotlib.pyplot as plt
import time
from pathlib import Path


def topKAccuracy(guess, target, rank=1):
    sorted_guess, indices = torch.sort(guess)
    hits = 0.0
    for i in range(len(sorted_guess)):
        for k in range(rank):
            if target[i] == indices[i][k]:
                hits += 1
    return round((hits/len(guess)) * 100, 2)


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


def loadJson(path, ablate=0):
    start_time = time.time()
    # leggo i json dal path e ritono i tensor di local, global e target
    loc = torch.empty([392, 750, 188], dtype=torch.float)
    glob = torch.empty([392, 1])
    targ = torch.empty((392), dtype=torch.long)
    index = 0
    for i in range(3):
        with open(path + "data" + str(i) + ".json") as file:
            data = json.load(file)
        for anim_i, anim in enumerate(data["Items"]):  # 392
            # local
            for frame_i, frame in enumerate(anim["frames"]):  # 750
                temp = []
                if ablate == 0:
                    temp += frame.pop("positions")
                    temp += frame.pop("accelerations")
                    temp += frame.pop("velocities")
                    temp += frame.values()
                elif ablate == 1:
                    temp.append(frame["hunchback"])
                    temp.append(frame["outToeingL"])
                    temp.append(frame["outToeingR"])
                elif ablate == 2:
                    temp += frame["positions"]
                    temp += frame["accelerations"]
                    temp += frame["velocities"]
                elif ablate == 3:
                    temp.append(frame["bodyOpennessU"])
                    temp.append(frame["bodyOpennessL"])
                    temp.append(frame["bctU"])
                    temp.append(frame["bctL"])
                    temp.append(frame["bctF"])
                if (loc.shape != torch.Size([392, 750, len(temp)])):
                    loc, _ = torch.split(
                        loc, [len(temp), 188-len(temp)], dim=2)
                loc[index][frame_i] = torch.tensor(temp)
            # global
            glob[index] = torch.tensor(anim["mediaLungPass"])
            # target
            targ[index] = anim["index"]
            index += 1
    print("\n--- JSON loaded in %s seconds ---" % (time.time() - start_time))
    return loc, glob, targ
