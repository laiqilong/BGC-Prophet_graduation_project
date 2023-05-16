#!/usr/bin/env python3
import torch as pt
import pandas as pd

from torch.utils.data import DataLoader

from utils import *
from data import *
from model import LSTMNET

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_,_,_,x = read_file('data/training_label.txt','data/training_nolabel.txt','data/testing_data.txt')
requires_grad = False
sen_len = 20
batch_size = 256

prepocess = DataPreprocess(x, sen_len, w2vmodel='./word2vec.sav')
embedding = prepocess.make_embedding()
data_x = prepocess.sentence_word2idx()
test_dataset = TwitterDataset(data_x, None)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
print('Testing loader prepared.')

model  = pt.load('ckpt.sav')
model.eval()
results = []

with pt.no_grad():
  for i,item in enumerate(test_loader):
    item = item.to(device, dtype=pt.long)
    outputs = model(item)
    outputs = outputs.squeeze()
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    results += outputs.int().tolist()

tmp = pd.DataFrame({"id": [str(i) for i in range(len(x))], "label": results})
print("Saving csv ...")
tmp.to_csv('predict.csv', index=False)
print("Predicting finished.")
