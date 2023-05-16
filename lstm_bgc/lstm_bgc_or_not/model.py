
#!/usr/bin/env python3
import torch as pt

class LSTMNET(pt.nn.Module):
  def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=True):
    super(LSTMNET, self).__init__()
    self.embedding = pt.nn.Embedding(embedding.size(0), embedding.size(1))
    self.embedding.weight = pt.nn.Parameter(embedding, requires_grad=requires_grad)
    self.LSTM = pt.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
    self.comp = pt.nn.Sequential(
      pt.nn.Dropout(dropout),# (batch_size, max_len, 2*hidden_dim)
      pt.nn.Linear(2*hidden_dim, 64), #(batch_size, max_len, 64)
      pt.nn.Tanh()
      pt.nn.Linear(64, 1) #(batch_size, max_len, 1)
      pt.nn.Sigmoid())

  def forward(self, inputs):
    inputs = self.embedding(inputs) #(batch_size, lenOfBGC, emb_dimen)
    x, _ = self.LSTM(inputs, None)#取所有时间步的输出张量，忽略隐藏状态张量
    # x = x[:,-1,:]#取最后一个时间步输出
    x = self.comp(x)
    return(x)
