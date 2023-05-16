#!/usr/bin/env python3
import torch as pt

class LSTMNET(pt.nn.Module):
  def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=True):
    super(LSTMNET, self).__init__()
    self.embedding = pt.nn.Embedding(embedding.size(0), embedding.size(1))
    self.embedding.weight = pt.nn.Parameter(embedding, requires_grad=requires_grad)
    # 将嵌入层包含辱LSTM，输入为词序号
    self.LSTM = pt.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
    self.TD = TimeDistributed(FeatLayer(hidden_dim*2, 1), bacth_first=True)
    self.comp = pt.nn.Sequential(
      pt.nn.Dropout(dropout),
      pt.nn.Linear(2*hidden_dim, 1),
      pt.nn.Sigmoid())

  def forward(self, inputs):
    inputs = self.embedding(inputs) #(batch_size, lenOfBGC, emb_dimen)
    x, _ = self.LSTM(inputs, None)#取所有时间步的输出张量，忽略隐藏状态张量
    x = x[:,-1,:]#取最后一个时间步输出(batch_size, 1, 2*hidden_dim)
    y = self.TD(x)
    x = self.comp(x) # 同时输出BGC种类和每个BGC的类别？
    return(x,y)

class FeatLayer(pt.nn.Module):
  def __init__(self, dim_in, dim_out):
    super(FeatLayer, self).__init__()
    sizes = []
    # 四转一？如dim_in = 64
    while dim_in > 4:
       sizes.append(dim_in)
       dim_in//=4
       # sizes: [64, 16, 4]逐层降采样？最后变为一维
    self.feat = pt.nn.Sequential(
        *[pt.nn.Linear(dim_i, dim_i // 4) for dim_i in sizes],
        pt.nn.Linear(sizes[-1], dim_out),
        pt.nn.Sigmoid()
        )

    def forward(self, x):
      return(self.feat(x))

class TimeDistributed(nn.Module):
  def __init__(self, module, batch_first=False):
      super(TimeDistributed, self).__init__()
      self.module = module
      self.batch_first = batch_first


  def forward(self, x):
      # (batch_size, 1, 2*hidden_dim)
      if len(x.size()) <= 2:
          return self.module(x)
      # reshape input data --> (samples * timesteps, input_size)
      # squash timesteps
      x_reshaped = x.contiguous().view(-1, x.size(-1))
      y = self.module(x_reshaped)
      # We have to reshape Y
      if self.batch_first:
          # (samples, timesteps, output_size)
          y = y.contiguous().view(x.size(0), -1, y.size(-1))
      else:
          # (timesteps, samples, output_size)
          y = y.contiguous().view(-1, x.size(1), y.size(-1))
      return y

