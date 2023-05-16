import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import random

# np.random.seed(42)
# random.seed(42)

class DataReader:

    def __init__(self, modelPath, datasetPath, max_len, test_ratio = 0.2, left_padding=False) -> None:
        self.modelPath = modelPath
        self.datasetPath = datasetPath
        self.max_len = max_len
        self.word2vecModel = Word2Vec.load(self.modelPath)
        self.embedding_dim = self.word2vecModel.wv.vector_size
        self.padding = np.zeros(self.embedding_dim)
        self.left_padding = left_padding
        self.test_ratio = test_ratio
        self.__getData()
        self.__train_test_split()

    def __getData(self):
        self.df = pd.read_csv(self.datasetPath)
        # 读取csv并展开为list便于处理
        # self.df['sentence'] = self.df['sentence'].apply(lambda x:x.split())
        self.df['labels'] = self.df['labels'].apply(lambda x:x.split())
        self.df['TDsentence'] = self.df['TDsentence'].apply(lambda x:x.split())
        self.df['TDlabels'] = self.df['TDlabels'].apply(lambda x:x.split()).apply(lambda x:[int(i) for i in x])
        self.df['labels_rep'] = self.df['labels'].apply(lambda x:x[0]) # 仅取第一个便于分层抽样
        # self.df['length'] = self.df['sentence'].apply(lambda x:len(x))
        # self.df = self.df.drop(self.df[self.df['length']<=1].index)
        # self.df = self.df.reset_index(drop=True)
        # labels_set = map(set, self.df['labels'])
        # self.labels_list = list(set().union(*list(labels_set)))
        self.labels_list = ['Alkaloid', 'Terpene', 'NRP', 'Polyketide', 'RiPP', 'Saccharide', 'Non', 'Other']
        print(self.labels_list)
        self.labels_num = len(self.labels_list)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # print(idx)
        # print(self.df)
        # print(self.df['sentence'])
        sentence = self.df['TDsentence'][idx]
        labels = self.df['labels'][idx]
        TDlabels = self.df['TDlabels'][idx]

        sentence_embedding = []
        for i in range(len(sentence)):
            word = sentence[i]
            if word in self.word2vecModel.wv:
                embedding = self.word2vecModel.wv[word]
                sentence_embedding.append(embedding)
            else:
                sentence_embedding.append(self.padding)
                TDlabels[i] = 0

        if len(sentence_embedding) == self.max_len:
            pass
        elif len(sentence_embedding)>self.max_len:
            sentence_embedding = sentence_embedding[:self.max_len]
        else:
            # left padding
            if self.left_padding:
                sentence_embedding = [self.padding] * (self.max_len - len(sentence_embedding)) + sentence_embedding
                TDlabels = [0]*(self.max_len-len(sentence_embedding)) + TDlabels
            else:
                sentence_embedding += [self.padding] * (self.max_len - len(sentence_embedding))
                TDlabels += [0] * (self.max_len - len(sentence_embedding))

        sentence_embedding = np.array(sentence_embedding)

        if labels == ['Unknown']:
            labels = random.choice(self.labels_list)
        # 在labels未知时随机一个值以便后面eval时能计算
        labels_onehot = np.array([1 if label in labels else 0 for label in self.labels_list])

        TDlabels = np.array(TDlabels)
        # print(sentence_embedding)
        # print(labels_onehot)
        return sentence_embedding, labels_onehot, TDlabels
    
    def __train_test_split(self):
        groups = self.df.groupby('labels_rep')
        # print(self.df['labels_rep'])
        # print(groups)
        self.train = []
        self.test = []
        # 根据代表的类别进行分层抽样
        for k, v in groups.groups.items():
            # k为代表的标签
            # v为索引
            vv = list(v)
            # print(vv)
            random.shuffle(vv)
            self.train += vv[int(v.size*self.test_ratio):]
            self.test += vv[:int(v.size*self.test_ratio)]
        # print(len(self.train))

    
class BGCLabelsDataset(Dataset):
    def __init__(self, data, mode='train') -> None:
        super().__init__()
        self.data = data
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            # print(len(self.data.train))
            return len(self.data.train)
        elif self.mode == 'test':
            # print(len(self.data.test))
            return len(self.data.test)
        elif self.mode == 'eval':
            return len(self.data)
        else:
            raise ValueError("No such mode: {}!".format(self.mode))
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            index = self.data.train[idx]
        elif self.mode == 'test':
            index = self.data.test[idx]
        elif self.mode == 'eval':
            index = idx
        else:
            raise ValueError("No such mode: {}!".format(self.mode))
        
        sentence_embedding, labels_onehot, TDlabels = self.data[index]
        assert len(sentence_embedding)==len(TDlabels), f'{idx} sample: length of sentence: {len(sentence_embedding)}, length of labels: {len(TDlabels)}'
        # print(index, self.data.df.iloc[index], sentence_embedding.shape, labels_onehot.shape, TDlabels.shape)
        return torch.tensor(sentence_embedding, dtype=torch.float32), torch.tensor(labels_onehot, dtype=torch.float32), torch.tensor(TDlabels, dtype=torch.float32)





        
            
            




