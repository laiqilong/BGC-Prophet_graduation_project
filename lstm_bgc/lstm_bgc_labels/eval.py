import torch
import argparse
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from wisedata import DataReader, BGCLabelsDataset
from model import LSTMNET
from loss import LSTMLoss
from utils import evaluate

from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class LSTMeval:

    def __init__(self, args, data, model, loss) -> None:
        self.args = args

        self.data = data
        self.model = model
        self.loss = loss

        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.lstmPath = args.lstmPath
        self.results = []
        self.labels = []
        
        self.model = torch.load(self.lstmPath)
        self.dataset = BGCLabelsDataset(data, mode='test')
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        os.makedirs(self.save_dir, exist_ok=True)

    def eval(self):
        self.model.eval()
        total_acc = 0
        total_loss = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.dataLoader),desc='Evaluate', leave=True):
                sentence, labels = data[0], data[1]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sentence) # (batch_size, labels_num)
                # loss = self.loss(outputs, labels)
                # total_loss += loss.item()

                correct = evaluate(outputs.clone().detach(), labels)/self.data.labels_num
                accuray = correct*100/torch.sum(labels).item()
                self.labels.append(labels.numpy(force=True))
                self.results.append(outputs.numpy(force=True)) # (batch_size, labels_num)

                total_acc += accuray
        self.total_acc = total_acc/len(self.dataLoader)
        self.results = np.vstack(self.results)
        self.labels = np.vstack(self.labels)
        # self.total_loss = total_loss/len(self.dataLoader)

        print('#Acc:%.3f' % self.total_acc)

    def saveResults(self):
        # with open(self.save_dir + 'lstm_results.', 'w') as fp:
        fpr, tpr, thresholds = roc_curve(self.labels.flatten(), self.results.flatten())
        roc_auc = auc(fpr, tpr)
        print('roc_auc: ', roc_auc)
        cm = confusion_matrix(self.labels.flatten(), self.results.flatten()>0.5)
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        print('recall: ', tp/(fn+tp))
        print('precision: ', tp/(fp+tp))
        np.save(self.save_dir + 'lstm_results_test.npy', self.results,allow_pickle=True)
        np.save(self.save_dir + 'lstm_labels_test.npy', self.labels, allow_pickle=True)

            



if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'LSTM',
        description='LSTM model to predict labels',
    )

    parser.add_argument('--modelPath', required=True)
    parser.add_argument('--datasetPath', required=True)
    parser.add_argument('--lstmPath', required=True)
    # parser.add_argument('--name', required=True)
    # parser.add_argument('--modelSavePath', required=False)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--left_padding', '-l', action='store_true', required=False)
    parser.add_argument('--hidden_dim', required=True, type=int)
    parser.add_argument('--num_layers', required=False, default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', required=True, type=int)
    # parser.add_argument('--learning_rate', required=True, type=float)
    # parser.add_argument('--interval', required=False, default=10, type=int)
    # parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--save_dir', default='./resultSave/')
    parser.add_argument('--labels_num', default=8, type=int)
    # parser.add_argument()

    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    data = DataReader(args.modelPath, args.datasetPath, args.max_len, left_padding=args.left_padding)
    embedding_dim = data.embedding_dim
    if args.lstmPath:
        model = torch.load(args.lstmPath)
    else:
        print('Error: no model path')
        exit(0)
        model = LSTMNET(embedding_dim, args.hidden_dim, num_layers=args.num_layers, max_len=args.max_len, labels_num=args.labels_num, dropout=args.dropout)
    loss = LSTMLoss()

    evaluater = LSTMeval(args=args, data=data, model=model, loss=loss)
    evaluater.eval()
    evaluater.saveResults()
