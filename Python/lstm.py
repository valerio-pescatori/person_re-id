import torch
import torch.nn as nn
import json
from pathlib import Path


data = None
with open(str(Path.cwd())+ "\data.json", "r") as file:
    data = json.load(file)

data = data["Items"]
# data è una lista di oggetti (dizionari), ogni dict rappresenta un'animazione


# architettura nn:
# 1  lstm layer x local mf
# concateno l'output del lstm layer con le global mf
# 1 dense layer che mi restituisce l'indice dell'animazione

# preparo i dati

# input del primo layer sono le features locali quindi
# body openness upper
# body openness lower
# bct upper
# bct lower
# bct full
# out-toeing l
# out-toeing r
# hunchback
# TOT: 8
# N.b.: sono 8 valori per ogni frame, ma ogni animazione ha 
#       un numero variabile di frames, (comunque nell'ordine delle centinaia)

# input del 1° dense layer è composto da:
# output del layer precedente + vettore delle features globali (solo mediaLungPass)


# Hyper-parameters

input_size = 8*750
hidden_size = 64 # arbitrario
num_classes = 56 # num di animazioni
num_epochs = 3
batch_size = 10
learning_rate = 0.001


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        # definisco la struttura
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.dense = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        lstm_out, hidden = self.lstm(input.view(len(input), ))
        pass

