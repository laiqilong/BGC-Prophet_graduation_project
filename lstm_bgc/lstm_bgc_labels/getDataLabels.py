import torch
import numpy as np
import pandas

from wisedata import DataReader


if __name__=='__main__':
    modelPath = '/home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav'
    datasetPath = './data/BGC_labels_dataset.csv'
    max_len = 64
    data = DataReader(modelPath, datasetPath, max_len)
    print(data.labels_list)
