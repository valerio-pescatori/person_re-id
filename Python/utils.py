import json
from pathlib import Path


def preprocessData(file_path):
    localf = []
    globalf = []
    target = []
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
        localf.append(anim_local_features)
        globalf.append([animation["mediaLungPass"]])
        target.append(animation["index"])
    return localf, globalf, target


def loadJson(loc, glob, targ):
    # divido i samples tra training e testing
    # 3 per training 4 per testing
    for i in range(3):
        t = preprocessData(str(Path.cwd().parent) +
                           "\\Data\\data" + str(i) + ".json")
        loc += t[0]
        glob += t[1]
        targ += t[2]
    return loc, glob, targ
