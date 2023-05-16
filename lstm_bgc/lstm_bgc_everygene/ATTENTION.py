#!/usr/bin/env python3
import torch as pt
import numpy as np
import torch.nn.functional as F

class ATTENTION(pt.nn.Module):
  def __init__(self, dim_input, dim_k=None, dim_v=None):
    super(ATTENTION, self).__init__()
    self.dim_input = dim_input
    if(dim_k is None):
      dim_k = dim_input
    if(dim_v is None):
      dim_v = dim_input
    self.W_q = pt.nn.Linear(dim_input, dim_k, bias=False)
    self.W_k = pt.nn.Linear(dim_input, dim_k, bias=False)
    self.W_v = pt.nn.Linear(dim_input, dim_v, bias=False)

    self._norm_fact = 1 / np.sqrt(dim_k)

  def forward(self, x):
    Q,K,V = self.W_q(x), self.W_k(x), self.W_v(x)
    KT = K.permute(0, 2, 1)
    atten = pt.nn.Softmax(dim=-1)(pt.bmm(Q, KT) * self._norm_fact)
    return(pt.bmm(atten, V)) 

class ATTBlock(pt.nn.Module):
  def __init__(self, dim_input, depth=4):
    super(ATTBlock, self).__init__()
    layers = [ATTENTION(dim_input) for i in range(depth)]
    self.block = pt.nn.Sequential(*layers)

  def forward(self, x):
    return(self.block(x))

#class ATTLSTMNET(pt.nn.Module):
#  def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, depth=4, requires_grad=True):
#    super(ATTLSTMNET, self).__init__()
#    self.embedding = pt.nn.Embedding(embedding.size(0), embedding.size(1))
#    self.embedding.weight = pt.nn.Parameter(embedding, requires_grad=requires_grad)
#    self.attention = ATTBlock(embedding_dim, depth=depth)
#    self.LSTM = pt.nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=num_layers, batch_first=True)
#    self.comp = pt.nn.Sequential(
#      pt.nn.Dropout(dropout),
#      pt.nn.Linear(hidden_dim*2, 1),
#      pt.nn.Sigmoid())
#
#  def forward(self, inputs):
#    inputs = self.embedding(inputs)
#    inputs_att = self.attention(inputs)
#    x, (hn,cn) = self.LSTM(inputs, None)
#    xx = self.comp(x[:,-1,:])
#    return(xx)

