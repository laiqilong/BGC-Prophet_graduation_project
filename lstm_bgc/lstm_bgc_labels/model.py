
#!/usr/bin/env python3
import torch 
from torch import nn

class LSTMNET(torch.nn.Module):

  def __init__(self, embedding_dim, hidden_dim, num_layers, max_len, labels_num, dropout=0.5):
    super(LSTMNET, self).__init__()
    # self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
    # self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)
    self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
    self.comp = nn.Sequential(
      nn.Dropout(dropout),# (batch_size, max_len, 2*hidden_dim)
      nn.Linear(2*hidden_dim, 64), #(batch_size, max_len, 64)
      nn.Tanh(),
      nn.Linear(64, 1), #(batch_size, max_len, 1)
      nn.Sigmoid())
    
    self.out = nn.Sequential(
      nn.Linear(max_len, labels_num), 
      nn.Sigmoid()
    )

  def forward(self, inputs):
     #(batch_size, lenOfBGC, emb_dimen)
    x, _ = self.LSTM(inputs, None)#取所有时间步的输出张量，忽略隐藏状态张量
    # x = x[:,-1,:]#取最后一个时间步输出
    x = self.comp(x)
    x = self.out(x.squeeze())
    return(x) # (batch_size, labels_num)
