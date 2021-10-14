import torch
import json
from pathlib import Path
from pprint import pprint


data = None
with open(str(Path.cwd())+ "\data.json", "r") as file:
    data = json.load(file)

data = data["Items"]
# data è una lista di oggetti (dizionari), ogni dict rappresenta un'animazione


