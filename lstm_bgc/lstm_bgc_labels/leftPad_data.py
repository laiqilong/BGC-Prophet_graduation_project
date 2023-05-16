#!/usr/bin/env python3
import torch as pt
import numpy as np

from gensim.models import Word2Vec

class DataPreprocess:
  def __init__(self, sentences, max_len, w2vmodel='./word2vec.sav'):
    self.sentences = sentences
    self.sen_len = max_len
    self.w2v_path = w2vmodel
    self.index2word = []
    self.word2index = {}
    self.embedding_matrix = []
    self.embedding = Word2Vec.load(self.w2v_path)
    self.embedding_dim = self.embedding.vector_size

  def add_embedding(self, word):
    vector = pt.empty(1, self.embedding_dim)
    pt.nn.init.uniform_(vector)
    self.word2index[word] = len(self.word2index)
    self.index2word.append(word)
    self.embedding_matrix =  pt.cat([self.embedding_matrix, vector], 0)

  def make_embedding(self):
    for i,word in enumerate(self.embedding.wv.key_to_index):
      self.word2index[word] = len(self.word2index)
      self.index2word.append(word)
      self.embedding_matrix.append(self.embedding.wv[word])
    self.embedding_matrix = np.array(self.embedding_matrix)
    self.embedding_matrix = pt.tensor(self.embedding_matrix)
    self.add_embedding("<PAD>")
    return(self.embedding_matrix)

  def pad_sequence(self, sentence):
    if(len(sentence) > self.sen_len):
      sentence = sentence[:self.sen_len]
    else:
      pad_len = self.sen_len - len(sentence)
      for _ in range(pad_len):
        sentence.insert(0, self.word2index["<PAD>"])
    assert(len(sentence) == self.sen_len)
    return(sentence)

  def sentence_word2idx(self):
    sentence_list = []
    for i,item in enumerate(self.sentences):
      sentence_index = []
      for word in item:
        if(word in self.word2index.keys()):
          sentence_index.append(self.word2index[word])
        else:
          continue
      sentence_index = self.pad_sequence(sentence_index)
      sentence_list.append(sentence_index)
    return(pt.LongTensor(sentence_list))

  def labels2tensor(self, y):
    y = [int(label) for label in y]
    y = pt.LongTensor(y)
    return(y)

class BGCDataset(pt.utils.data.Dataset):
  def __init__(self, x, y):
    self.data = x
    self.label = y

  def __getitem__(self, index):
    if(self.label is None):
      return(self.data[index])
    return(self.data[index], self.label[index])

  def __len__(self):
    return(len(self.data))
