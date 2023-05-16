import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle
import numpy as np
import random

from wisedata import DataReader, BGCLabelsDataset
from model import LSTMTimeDistributedNet
from loss import *
from utils import evaluate



# 预测每个基因是否属于BGC
class LSTMTimeDistributedTrainer:
    def __init__(self, args, data, model, Labelloss, TimeDistributedLoss) -> None:
        self.args = args
        self.batch_size = args.batch_size
        self.interval = args.interval
        self.label_epochs = args.label_epochs
        self.ditribute_epochs = args.distribute_epochs
        self.save_dir = args.save_dir
        self.learning_rate = args.learning_rate
        # self.savePath = args.save_dir

        self.data = data
        self.train_dataset = BGCLabelsDataset(self.data, mode='train')
        self.test_dataset = BGCLabelsDataset(self.data, mode='test')
        self.train_dataLoader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        self.test_dataLoader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        self.model = model
        self.Labelloss = Labelloss
        self.TimeDistributedLoss = TimeDistributedLoss
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.save_path = self.save_dir + \
        f'lstm_model/bS_{self.batch_size}_lE_{self.label_epochs}_dE_{self.ditribute_epochs}_lR_{args.learning_rate}_mL_{args.max_len}_hD_{args.hidden_dim}_nL_{args.num_layers}_dP_{args.dropout}/'
        os.makedirs(self.save_path, exist_ok=True)
        # with open(self.save_path+'labels_list.pkl', 'wb') as fp:
        #     pickle.dump(self.data.labels_list, fp)
        #     print("Save labels_list")
        print(self.model)

    

    def train_label_step(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        for i, data in tqdm(enumerate(self.train_dataLoader),desc='Train labels', leave=True):
            sentence, labels, distribution = data[0], data[1], data[2]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            # distribution = distribution.to(self.device)

            self.optimizer.zero_grad()
            outputsLabels, outputsTD = self.model(sentence) # (batch_size, labels_num)
            # print("sentence:", sentence.data)
            # print("outputs:",outputs.data)
            # print("labels:",labels.data)
            # print(labels)
            loss = self.Labelloss(outputsLabels, labels)
            total_loss += loss.item()

            correct = evaluate(outputsLabels.clone().detach(), labels)
            # accuray = correct*100/labels.numel()
            accuray = correct*100/torch.sum(labels).item()
            total_acc += accuray

            loss.backward()
            self.optimizer.step()

            if (i%self.interval == 0):
                print('#Epoch:%d %d/%d loss:%.5f acc:%.3f' % (epoch, i, len(self.train_dataLoader), total_loss/self.interval, total_acc/self.interval))
                self.train_total_loss += total_loss
                self.train_total_acc +=total_acc
                total_loss = 0
                total_acc = 0
        self.train_total_acc /= len(self.train_dataLoader)
        self.train_total_loss /= len(self.train_dataLoader)

        torch.save(self.model, self.save_path + f'LSTM_Model_label_{epoch}.pt')

    def train_TD_step(self, epoch):
        self.model.train()
        total_label_loss = 0
        total_TD_loss = 0
        total_label_acc = 0
        total_TD_acc = 0
        for i, data in tqdm(enumerate(self.train_dataLoader), desc='Train TD', leave=True):
            sentence, labels, distribution = data[0], data[1], data[2]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            distribution = distribution.to(self.device)

            self.optimizer.zero_grad()
            outputsLabels, outputsTD = self.model(sentence)
            Labelloss = self.Labelloss(outputsLabels, labels)
            TDLoss = self.TimeDistributedLoss(outputsTD, distribution)
            total_label_loss += Labelloss
            total_TD_loss += TDLoss

            # 计算准确度
            Label_correct = evaluate(outputsLabels.clone().detach(), labels)
            TD_correct = evaluate(outputsTD.clone().detach(), distribution)
            Label_accuracy = Label_correct*100/torch.sum(labels).item()
            if torch.sum(distribution).item()<=0:
                if TD_correct<=0:
                    TD_accuracy=100
                else:
                    TD_accuracy=0
            else: 
                TD_accuracy = TD_correct*100/torch.sum(distribution).item()

            # Label_accuracy = Label_correct*100/labels.numel()
            # TD_accuracy = TD_correct*100/distribution.numel()

            total_label_acc += Label_accuracy
            total_TD_acc += TD_accuracy

            TDLoss.backward()
            self.optimizer.step()

            if (i%self.interval == 0):
                print('#Epoch:%d, %d/%d Loss:%.5f/%.5f lacc:%.3f dacc:%.3f' % (epoch, i, len(self.train_dataLoader), total_label_loss/self.interval, 
                    total_TD_loss/self.interval, total_label_acc/self.interval, total_TD_acc/self.interval))
                self.train_total_label_loss += total_label_loss
                self.train_total_label_acc += total_label_acc
                self.train_total_TD_loss += total_TD_loss
                self.train_total_TD_acc += total_TD_acc
                total_label_loss = 0
                total_TD_loss = 0
                total_label_acc = 0
                total_TD_acc = 0
        self.train_total_label_acc /= len(self.train_dataLoader)
        self.train_total_label_loss /= len(self.train_dataLoader)
        self.train_total_TD_acc /= len(self.train_dataLoader)
        self.train_total_TD_loss /= len(self.train_dataLoader)
        torch.save(self.model, self.save_path + f'LSTM_Model_TD_{epoch}.pt')

    def validate_step(self):
        self.model.eval()
        with torch.no_grad():
            total_label_loss = 0
            total_TD_loss = 0
            total_label_acc = 0
            total_TD_acc = 0
            for i, data in tqdm(enumerate(self.test_dataLoader), desc="Test", leave=True):
                # 数据预处理
                sentence, labels, distribution = data[0], data[1], data[2]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                distribution = distribution.to(self.device)

                # 模型推理，计算loss
                outputsLabels, outputsTD = self.model(sentence)
                Labelloss = self.Labelloss(outputsLabels, labels)
                TDLoss = self.TimeDistributedLoss(outputsTD, distribution)
                total_label_loss += Labelloss
                total_TD_loss += TDLoss

                # 计算准确度
                Label_correct = evaluate(outputsLabels.clone().detach(), labels)
                TD_correct = evaluate(outputsTD.clone().detach(), distribution)
                # Label_accuracy = Label_correct*100/labels.numel()
                # TD_accuracy = TD_correct*100/distribution.numel()
                Label_accuracy = Label_correct*100/torch.sum(labels).item()
                if torch.sum(distribution).item()<=0:
                    if TD_correct<=0:
                        TD_accuracy=100
                    else:
                        TD_accuracy=0
                else: 
                    TD_accuracy = TD_correct*100/torch.sum(distribution).item()
                total_label_acc += Label_accuracy
                total_TD_acc += TD_accuracy

        print('#Loss:%.5f/%.5f lacc:%.3f dacc:%.3f' % (total_label_loss/len(self.test_dataLoader), 
                                                       total_TD_loss/len(self.test_dataLoader), total_label_acc/len(self.test_dataLoader), total_TD_acc/len(self.test_dataLoader)))
        
        self.test_total_label_loss = total_label_loss / len(self.test_dataLoader)
        self.test_total_TD_loss = total_TD_loss / len(self.test_dataLoader)
        self.test_total_label_acc = total_label_acc / len(self.test_dataLoader)
        self.test_total_TD_acc = total_TD_acc / len(self.test_dataLoader)


    def train(self):
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, last_epoch=-1, verbose=True)  

        for epoch in range(self.label_epochs):
            self.train_total_loss = 0
            self.train_total_acc = 0
            self.train_label_step(epoch)
            # print(self.scheduler.get_last_lr())
            if self.scheduler.get_last_lr()[0]>0.00005:
                self.scheduler.step()
            print(f'Train Set: loss:{self.train_total_loss}, acc: {self.train_total_acc}')
            self.test_total_label_loss = 0
            self.test_total_TD_loss = 0
            self.test_total_label_acc = 0
            self.test_total_TD_acc = 0
            self.validate_step()
            print(f'Test Set: lloss:{self.test_total_label_loss}, lacc:{self.test_total_label_acc}')

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, last_epoch=-1, verbose=True)  

        for epoch in range(self.ditribute_epochs):
            self.train_total_label_acc = 0
            self.train_total_label_loss = 0
            self.train_total_TD_acc = 0
            self.train_total_TD_loss = 0
            self.train_TD_step(epoch=epoch)
            if self.scheduler.get_last_lr()[0]>0.00005:
                self.scheduler.step()
            print(f'Train Set: lloss:{self.train_total_label_loss}, lacc: {self.train_total_label_acc}')
            print(f'Train Set: tdloss:{self.train_total_TD_loss}, tdacc: {self.train_total_TD_acc}')
            self.test_total_label_loss = 0
            self.test_total_TD_loss = 0
            self.test_total_label_acc = 0
            self.test_total_TD_acc = 0
            self.validate_step()
            print(f'Test Set: lloss:{self.test_total_label_loss}, lacc:{self.test_total_label_acc}')
            print(f'Test Set: tdloss:{self.test_total_TD_loss}, tdacc:{self.test_total_TD_acc}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

            

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'LSTM',
        description='LSTM model to predict labels',
    )

    parser.add_argument('--modelPath', required=True)
    parser.add_argument('--datasetPath', required=True)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--left_padding', '-l', action='store_true', required=False)
    parser.add_argument('--hidden_dim', required=True, type=int)
    parser.add_argument('--num_layers', required=False, default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--learning_rate', required=True, type=float)
    parser.add_argument('--interval', required=False, default=10, type=int)
    parser.add_argument('--label_epochs', required=True, type=int)
    parser.add_argument('--distribute_epochs', required=True, type=int)
    parser.add_argument('--save_dir', default='./modelSave/')
    parser.add_argument('--load_label_model', action='store_true', required=False)
    parser.add_argument('--label_model_path', required=False)
    parser.add_argument('--two_gpu', required=False, action='store_true')
    parser.add_argument('--seed', required=False, default=42, type=int)
    # parser.add_argument()

    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    
    setup_seed(args.seed)
    data = DataReader(args.modelPath, args.datasetPath, args.max_len, left_padding=args.left_padding, test_ratio=0.2)
    embedding_dim = data.embedding_dim
    if args.load_label_model:
        model = torch.load(args.label_model_path)
    else:
        model = LSTMTimeDistributedNet(embedding_dim=embedding_dim, num_heads=args.num_heads, depth=args.depth, hidden_dim=args.hidden_dim, num_layers=args.num_layers, max_len=args.max_len, labels_num=data.labels_num, dropout=args.dropout)
    if args.two_gpu:
        model = torch.nn.DataParallel(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total}\nTrainable Parameters: {trainable}\n')
    # Labelloss = binary_focal_loss(alpha=0.25, gamma=2, reduction='sum')
    Labelloss = LSTMLoss()
    TimeDistributedLoss = FocalLoss(alpha=0.05, gamma=1, reduction='sum')
    # TimeDistributedLoss = LSTMLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)  
    trainer = LSTMTimeDistributedTrainer(args=args, data=data, model=model, Labelloss=Labelloss, TimeDistributedLoss=TimeDistributedLoss)
    trainer.train()