#!/usr/bin/env python3
import os
import sys

from gensim.models import word2vec
from utils import read_file

def train_word2vec(x):
  return(word2vec.Word2Vec(sentences=x, size = 300, window=5, min_count=5, sg=1, iter=20, alpha=2e-2,seed=1001, workers=12))

if(__name__ == '__main__'):
  print('#loading data...')
  ifn1,ifn2,ifn3,ofn = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
  x1,y1,x2,x3 = read_file(ifn1,ifn2,ifn3)
  print('#training word2vec and transforming to vectors by skip-gram...')
  print(x1[:3],x2[:3],x3[:3])
  model = train_word2vec(x1 + x2 + x3)
  model.save(ofn)
  print('#training done!')
  #rmodel = word2vec.Word2Vec.load('word2vec.sav')
