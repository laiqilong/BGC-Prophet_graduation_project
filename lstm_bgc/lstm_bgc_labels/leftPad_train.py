#!/usr/bin/env python3
import torch as pt
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from gensim.models.callbacks import CallbackAny2Vec
import datetime
import os
from multiprocessing import cpu_count

from utils import *
from leftpad_data import *
from model import *

start_time = datetime.datetime.now()
#参数设置
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if pt.cuda.is_available() else 'cpu'
requires_grad = False
sen_len = 128
batch_size = 64
epochs = 30
lr = 1e-4

loss_list=[]
class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 1

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_list.append(loss)
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

def train(train_loader, model, criterion, optimizer, epoch):
  model.train()#训练模式，使其可以进行前向传播和反向传播
  train_len = len(train_loader)
  total_loss, total_acc = 0, 0

  for i, (inputs,labels) in enumerate(train_loader):
    inputs = inputs.to(device, dtype=pt.long)
    #print(f'inputs:\n{inputs}')
    labels.unsqueeze(1)
    labels = labels.to(device, dtype=pt.float)
    optimizer.zero_grad()
    outputs = model(inputs)
    outputs = outputs.squeeze()

    loss =  criterion(outputs,labels)
    total_loss = total_loss + loss.item()

    correct = evaluate(outputs.clone().detach(), labels)
    total_acc += (correct / batch_size)
    loss.backward()
    optimizer.step()
    if(i % 10 == 0):
      print('#Epoch:%d\t%d/%d\tloss:%.5f\tacc:%.3f' % (epoch, i, train_len, loss.item(), correct*100/batch_size))

def validate(val_loader, model, criterion):
  model.eval()#进入评估模式，此时模型不会进行梯度计算，而是直接对输入数据进行前向传播，输出预测结果。
  val_len = len(val_loader)
  with pt.no_grad():
    total_loss, total_acc = 0,0
    for i, (inputs, labels) in enumerate(val_loader):
      inputs = inputs.to(device, dtype=pt.long)
      labels = labels.to(device, dtype=pt.float)
      outputs = model(inputs)
      print(outputs)
      outputs = outputs.squeeze()
      loss =  criterion(outputs,labels)
      total_loss += loss.item()
      correct = evaluate(outputs,labels)
      total_acc += (correct/batch_size)
    print('#Loss:%.5f\tacc:%.3f' % (total_loss/val_len, total_acc/val_len*100))
  return(total_acc/val_len*100)

#设置最大线程
cpu_num = cpu_count() # 自动获取最大核心数目
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

data_x, data_y = read_file(label='/home/yaoshuai/tools/lstm_bgc/data/training_label.txt')
preprocess = DataPreprocess(data_x, sen_len, w2vmodel='/home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav')
embedding = preprocess.make_embedding()
print(embedding.shape)
data_x = preprocess.sentence_word2idx()
data_y = preprocess.labels2tensor(data_y)

x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2, random_state=5, shuffle=True, stratify=data_y)

train_dataset = BGCDataset(x_train,y_train)
val_dataset = BGCDataset(x_test,y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = LSTMNET(embedding, embedding_dim=200, hidden_dim=256, num_layers=1, dropout=0.5, requires_grad=requires_grad).to(device)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
best_acc = 0

for epoch in range(epochs):
  train(train_loader, model, criterion, optimizer, epoch)
  print('validate set:')
  total_acc = validate(val_loader, model, criterion)
  print('train set:')
  total_acc_train = validate(train_loader, model, criterion)
  if(total_acc > best_acc):
     best_acc = total_acc
     pt.save(model, 'step1_leftpad.sav')

end_time = datetime.datetime.now()
print('总时间：', (end_time-start_time))
