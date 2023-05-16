import torch
import argparse
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import pickle
import random

from torch.utils.tensorboard import SummaryWriter   
from wisedata import DataReader, BGCLabelsDataset
from model import transformerEncoderLabelsNet
from loss import trainLoss
from utils import evaluate
from final_val import final_validation

class transformerEncoderLabelseval:

    def __init__(self, args, writer, data, model, loss) -> None:
        self.args = args
        self.writer = writer
        self.data = data
        self.model = model
        self.loss = loss

        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.transformerEncoderPath = args.transformerEncoderPath
        self.name = args.name
        self.results = []
        self.labels = []
        
        # self.model = torch.load(self.lstmPath)
        self.model = model
        self.dataset = BGCLabelsDataset(data, mode='test')
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        os.makedirs(self.save_dir, exist_ok=True)

    def eval(self):
        self.model.eval()
        total_labels_loss = 0
        total_labels_acc = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.dataLoader), desc='Evaluate', leave=True):
                sentence, labels, distribution = data[0], data[1], data[2]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                distribution = distribution.to(self.device)

                outputsLabels = self.model(sentence, distribution) 
                # loss = self.loss(outputs, labels)
                # total_loss += loss.item()
                # labelloss = self.loss(outputsLabels, labels)
                labelsLoss = self.loss(outputsLabels, labels)
                # total_label_loss += labelloss
                total_labels_loss += labelsLoss

                # Label_correct = evaluate(outputsLabels.clone().detach(), labels)
                # Label_accuracy = Label_correct*100/torch.sum(labels).item()
                labels_correct = evaluate(outputsLabels.clone().detach(), labels)
                if torch.sum(labels).item()!=0:
                    labels_accuracy = labels_correct*100/torch.sum(labels).item()
                else:
                    predict = outputsLabels.clone().detach()
                    predict[predict>0.5] = 1.
                    predict[predict<0.5] = 0.
                    labels_accuracy = torch.sum(torch.eq(predict, labels)).item()/labels.numel()


                # total_label_acc += Label_accuracy
                total_labels_acc += labels_accuracy
                if (i%10 == 0):
                    print(f'{i}/{len(self.dataLoader)} labels_accuracy: {labels_accuracy}')
                self.labels.append(labels.numpy(force=True))
                self.results.append(outputsLabels.numpy(force=True)) 

            
        # self.total_label_acc = total_label_acc/len(self.dataLoader)
        self.total_labels_acc = total_labels_acc/len(self.dataLoader)
        self.labels = np.vstack(self.labels).flatten()
        self.results = np.vstack(self.results).flatten()
        print(f'labels.shape: {self.labels.shape}')
        print(f'result.shape: {self.results.shape}')

        # self.total_loss = total_loss/len(self.dataLoader)

        print('#labels_acc:%.3f' % self.total_labels_acc)

    def saveResults(self):
        # with open(self.save_dir + 'lstm_results.', 'w') as fp:
        np.save(self.save_dir + f'{self.name}_outputsLabels.npy', self.results, allow_pickle=True)
        np.save(self.save_dir + f'{self.name}_labels.npy', self.labels, allow_pickle=True)
        self.writer.add_pr_curve('pr_curve', self.results, self.labels, 0)
        self.plot = final_validation(x_true=self.labels, x_pred=self.results, outpath=self.save_dir + self.name)
        self.plot.result()

            
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
    # parser.add_argument('--modelPath', required=True)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--left_padding', '-l', action='store_true', required=False)
    parser.add_argument('--nhead', type=int, required=True)
    parser.add_argument('--num_encoder_layers', required=True, default=4, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', required=True, type=int)
    # parser.add_argument('--learning_rate', required=True, type=float)
    # parser.add_argument('--interval', required=False, default=10, type=int)
    # parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--save_dir', default='./resultSave/')
    # parser.add_argument('--labels_num', default=8, type=int)
    parser.add_argument('--transformerEncoderPath', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seed', required=False, default=42, type=int)
    # parser.add_argument()

    args = parser.parse_args()
    setup_seed(args.seed)

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    writer = SummaryWriter('./log/TransformerEncoder/')
    data = DataReader(args.modelPath, args.datasetPath, args.max_len, left_padding=args.left_padding)
    embedding_dim = data.embedding_dim
    # model = LSTMTimeDistributedNet(embedding_dim, args.hidden_dim, num_layers=args.num_layers, max_len=args.max_len, labels_num=args.labels_num, dropout=args.dropout)
    model = torch.load(args.transformerEncoderPath)
    loss = trainLoss()

    evaluater = transformerEncoderLabelseval(args=args, writer=writer, data=data, model=model, loss=loss)
    evaluater.eval()
    evaluater.saveResults()
