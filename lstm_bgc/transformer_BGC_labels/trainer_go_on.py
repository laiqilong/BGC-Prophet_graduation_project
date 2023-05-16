import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter   
from tqdm import tqdm
import os
import pickle
import numpy as np
import random
import shutil

from wisedata import DataReader, BGCLabelsDataset
from model import transformerEncoderLabelsNet
from loss import trainLoss
from utils import evaluate



# 预测每个基因是否属于BGC
class TransformerEncoderLabelsGoOnTrainer:
    def __init__(self, args, writer, data, model, Labelloss) -> None:
        self.args = args
        self.writer = writer
        self.d_model = data.embedding_dim
        self.batch_size = args.batch_size
        self.interval = args.interval
        self.label_epochs = args.label_epochs
        # self.ditribute_epochs = args.distribute_epochs
        self.save_dir = args.save_dir
        self.learning_rate = args.learning_rate
        # self.savePath = args.save_dir

        self.data = data
        self.train_dataset = BGCLabelsDataset(self.data, mode='train')
        print('Train Data length: ', len(self.train_dataset))
        self.test_dataset = BGCLabelsDataset(self.data, mode='test')
        print('Train Data length: ', len(self.test_dataset))
        self.train_dataLoader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        self.test_dataLoader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        self.model = model
        self.Labelloss = Labelloss
        # self.TimeDistributedLoss = TimeDistributedLoss
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.save_path = self.save_dir + \
        f'transformerEncoder_labels/bS_{self.batch_size}_lE_{self.label_epochs}_lR_{args.learning_rate}_mL_{args.max_len}_d_{self.d_model}_nEL_{args.num_encoder_layers}_dP_{args.dropout}/'
        os.makedirs(self.save_path, exist_ok=True)
        # with open(self.save_path+'labels_list.pkl', 'wb') as fp:
        #     pickle.dump(self.data.labels_list, fp)
        #     print("Save labels_list")
        print(self.model)
        # self.writer.add_graph(model=self.model, input_to_model=torch.randn(self.batch_size, args.max_len, data.embedding_dim))

    

    def train_label_step(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        for i, data in tqdm(enumerate(self.train_dataLoader),desc='Train labels', leave=True):
            sentence, labels, distribution = data[0], data[1], data[2]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            distribution = distribution.to(self.device)

            self.optimizer.zero_grad()
            outputsLabels = self.model(sentence, distribution) # (batch_size, labels_num)
            loss = self.Labelloss(outputsLabels, labels)
            total_loss += loss.item()

            correct = evaluate(outputsLabels.clone().detach(), labels)
            # accuray = correct*100/labels.numel()
            accuray = correct*100/torch.sum(labels).item()
            total_acc += accuray

            loss.backward()
            self.optimizer.step()

            if (i!=0 and i%self.interval == 0):
                print('#Epoch:%d %d/%d loss:%.5f acc:%.3f' % (epoch, i, len(self.train_dataLoader), total_loss/self.interval, total_acc/self.interval))
                self.train_total_loss += total_loss
                self.train_total_acc +=total_acc
                total_loss = 0
                total_acc = 0
        self.train_total_acc /= len(self.train_dataLoader)
        self.train_total_loss /= len(self.train_dataLoader)

        torch.save(self.model, self.save_path + f'transformerEncoder_Model_labels_{2000+epoch}.pt')

    def validate_step(self):
        self.model.eval()
        with torch.no_grad():
            total_label_loss = 0
            total_label_acc = 0
            for i, data in tqdm(enumerate(self.test_dataLoader), desc="Test", leave=True):
                # 数据预处理
                sentence, labels, distribution = data[0], data[1], data[2]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                distribution = distribution.to(self.device)

                # 模型推理，计算loss
                outputsLabels = self.model(sentence, distribution)
                Labelloss = self.Labelloss(outputsLabels, labels)
                total_label_loss += Labelloss

                # 计算准确度
                Label_correct = evaluate(outputsLabels.clone().detach(), labels)
                # Label_accuracy = Label_correct*100/labels.numel()
                # TD_accuracy = TD_correct*100/distribution.numel()
                Label_accuracy = Label_correct*100/torch.sum(labels).item()
                total_label_acc += Label_accuracy

        print('#Loss:%.5f lacc:%.3f' % (total_label_loss/len(self.test_dataLoader), 
                                                       total_label_acc/len(self.test_dataLoader), ))
        
        self.test_total_label_loss = total_label_loss / len(self.test_dataLoader)
        self.test_total_label_acc = total_label_acc / len(self.test_dataLoader)

    def train(self):
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1, verbose=True)  

        for epoch in range(self.label_epochs):
            self.train_total_loss = 0
            self.train_total_acc = 0
            self.train_label_step(epoch)
            # tensorboard train
            self.writer.add_scalar('Loss/trainLabelsLoss', self.train_total_loss, epoch)
            self.writer.add_scalar('Acc/trainLabelsAcc', self.train_total_acc, epoch)
            # print(self.scheduler.get_last_lr())
            if epoch>2 and self.scheduler.get_last_lr()[0]>0.00005:
                self.scheduler.step()
            # if epoch>100 and self.scheduler.get_last_lr()[0]>0.00003:
            #     self.scheduler.step()
            # if epoch>300 and self.scheduler.get_last_lr()[0]>0.00002:
            #     self.scheduler.step()
            # if epoch>400 and self.scheduler.get_last_lr()[0]>0.00001:
            #     self.scheduler.step()
            print(f'Train Set: loss:{self.train_total_loss}, acc: {self.train_total_acc}')
            self.test_total_label_loss = 0
            self.test_total_TD_loss = 0
            self.test_total_label_acc = 0
            self.test_total_TD_acc = 0
            self.validate_step()
            # tensorboard validate
            self.writer.add_scalar('Loss/validateLabelsLoss', self.test_total_label_loss, epoch)
            self.writer.add_scalar('Acc/validateLabelsAcc', self.test_total_label_acc, epoch)
            print(f'Test Set: lloss:{self.test_total_label_loss}, lacc:{self.test_total_label_acc}')




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

            

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'transformerEncoder',
        description='transformerEncoder model to predict labels',
    )

    parser.add_argument('--modelPath', required=True)
    parser.add_argument('--datasetPath', required=True)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--left_padding', '-l', action='store_true', required=False)
    parser.add_argument('--goonModelPath', required=True,type=str)
    # parser.add_argument('--hidden_dim', required=True, type=int)
    parser.add_argument('--nhead', type=int, required=True)
    parser.add_argument('--num_encoder_layers', required=True, default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--concat_type', default='avg', type=str)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--learning_rate', required=True, type=float)
    parser.add_argument('--interval', required=False, default=10, type=int)
    parser.add_argument('--label_epochs', required=True, type=int)
    # parser.add_argument('--distribute_epochs', required=True, type=int)
    parser.add_argument('--save_dir', default='./modelSave/')
    parser.add_argument('--load_label_model', action='store_true', required=False)
    parser.add_argument('--two_gpu', required=False, action='store_true')
    parser.add_argument('--seed', required=False, default=42, type=int)
    # parser.add_argument()

    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    
    setup_seed(args.seed)
    # if os.path.exists('./log/TransformerEncoder_labels/'):
    #     # 删除之前log
    #     shutil.rmtree('./log/TransformerEncoder_labels/')
    writer = SummaryWriter('./log/TransformerEncoder_labels/')
    data = DataReader(args.modelPath, args.datasetPath, args.max_len, left_padding=args.left_padding)
    embedding_dim = data.embedding_dim
    if args.goonModelPath:
        model = torch.load(args.goonModelPath)
    else:
        print("Error!")
        print("Please input goonModelPath!")
        exit()
    if args.two_gpu:
        model = torch.nn.DataParallel(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total}\nTrainable Parameters: {trainable}\n')
    Labelloss = trainLoss()
    # TimeDistributedLoss = trainLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)  
    trainer = TransformerEncoderLabelsGoOnTrainer(args=args, writer=writer,data=data, model=model, Labelloss=Labelloss)
    trainer.train()