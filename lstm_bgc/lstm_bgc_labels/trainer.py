import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle

from wisedata import DataReader, BGCLabelsDataset
from model import LSTMNET
from loss import LSTMLoss
from utils import evaluate




class LSTMTrainer:
    def __init__(self, args, data, model, loss, optimizer, scheduler) -> None:
        self.args = args
        self.batch_size = args.batch_size
        self.interval = args.interval
        self.epochs = args.epochs
        self.save_dir = args.save_dir
        # self.savePath = args.save_dir

        self.data = data
        self.train_dataset = BGCLabelsDataset(self.data, mode='train')
        self.test_dataset = BGCLabelsDataset(self.data, mode='test')
        self.train_dataLoader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataLoader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.save_path = self.save_dir + f'{self.batch_size}_{self.epochs}_{args.learning_rate}_{args.max_len}_{args.hidden_dim}_{args.num_layers}_{args.dropout}/'
        os.makedirs(self.save_path, exist_ok=True)
        with open(self.save_path+'labels_list.pkl', 'wb') as fp:
            pickle.dump(self.data.labels_list, fp)
            print("Save labels_list")

    

    def train_step(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        for i, data in tqdm(enumerate(self.train_dataLoader),desc='Train', leave=True):
            sentence, labels = data[0], data[1]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sentence) # (batch_size, labels_num)
            # print("sentence:", sentence.data)
            # print("outputs:",outputs.data)
            # print("labels:",labels.data)
            # print(labels)
            loss = self.loss(outputs, labels)
            total_loss += loss.item()

            correct = evaluate(outputs.clone().detach(), labels)
            accuray = correct*100/torch.sum(labels).item()
            total_acc += accuray

            loss.backward()
            self.optimizer.step()

            if (i%self.interval == 0):
                print('#Epoch:%d\t%d/%d\tloss:%.5f\tacc:%.3f' % (epoch, i, len(self.train_dataLoader), total_loss/self.interval, total_acc/self.interval))
                self.train_total_loss += total_loss
                self.train_total_acc +=total_acc
                total_loss = 0
                total_acc = 0
        self.train_total_acc /= len(self.train_dataLoader)
        self.train_total_loss /= len(self.train_dataLoader)

        torch.save(self.model, self.save_path + f'LSTM_Model_{epoch}.pt')


    def validate_step(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        for i, data in tqdm(enumerate(self.test_dataLoader), desc="Test", leave=True):
            sentence, labels = data[0], data[1]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(sentence)
            loss = self.loss(outputs, labels)
            total_loss += loss

            correct = evaluate(outputs.clone().detach(), labels)
            accuracy = correct*100/torch.sum(labels).item()
            total_acc += accuracy

        print('#Loss:%.5f\tacc:%.3f' % (total_loss/len(self.test_dataLoader), total_acc/len(self.test_dataLoader)))
        self.test_total_loss = total_loss / len(self.test_dataLoader)
        self.test_total_acc = total_acc / len(self.test_dataLoader)
        # torch.save(self.model, )   

    def train(self):

        for epoch in range(self.epochs):
            self.train_total_loss = 0
            self.train_total_acc = 0
            self.train_step(epoch)
            print(self.scheduler.get_last_lr())
            if self.scheduler.get_last_lr()[0]>0.00005:
                self.scheduler.step()
            print(f'Train Set: loss:{self.train_total_loss}, acc: {self.train_total_acc}')
            self.test_total_acc = 0
            self.test_total_loss = 0
            self.validate_step()
            print(f'Test Set: loss:{self.test_total_loss}, acc:{self.test_total_acc}')

    


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'LSTM',
        description='LSTM model to predict labels',
    )

    parser.add_argument('--modelPath', required=True)
    parser.add_argument('--datasetPath', required=True)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--left_padding', '-l', action='store_true', required=False)
    parser.add_argument('--hidden_dim', required=True, type=int)
    parser.add_argument('--num_layers', required=False, default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--learning_rate', required=True, type=float)
    parser.add_argument('--interval', required=False, default=10, type=int)
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--save_dir', default='./modelSave/')
    # parser.add_argument()

    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    data = DataReader(args.modelPath, args.datasetPath, args.max_len, left_padding=args.left_padding)
    embedding_dim = data.embedding_dim
    model = LSTMNET(embedding_dim, args.hidden_dim, num_layers=args.num_layers, max_len=args.max_len, labels_num=data.labels_num, dropout=args.dropout)
    loss = LSTMLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)  
    trainer = LSTMTrainer(args=args, data=data, model=model, loss=loss, optimizer=optimizer, scheduler=scheduler)
    trainer.train()